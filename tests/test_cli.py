"""Tests for cjm_transcription_core.cli — parser smoke checks (no capabilities involved).

Projected from the cli notebook's parser-check cell at the golden-reference flip."""
import pytest

from cjm_transcription_core.cli import (build_parser, expand_sources,
                                         expand_sources_with_collections,
                                         parse_max_concurrent, parse_transcriber_spec)


def test_run_defaults_and_opt_ins():
    p = build_parser()
    args = p.parse_args(["run", "a.mp3", "b.mp3", "--yes", "--max-segment-duration", "120"])
    assert args.command == "run"
    assert args.audio == ["a.mp3", "b.mp3"]
    assert args.yes is True
    assert args.max_segment_duration == 120.0
    assert args.transcriber is None  # default applied in run_command (-> whisper)
    assert args.graph_capability is None  # emission is opt-in
    assert args.sysmon_capability is None  # Monitor is opt-in; default runs skip GPU attribution
    assert args.max_concurrent is None  # SG-33 overrides are opt-in (unset = queue default of 1)
    assert args.preprocessing_capability is None  # preprocessing is opt-in (default OFF)


def test_repeatable_transcriber_and_graph_flags():
    # repeatable --transcriber (stage-5 dual-transcriber run)
    p = build_parser()
    args = p.parse_args(["run", "a.mp3",
                         "--transcriber", "cjm-capability-whisper",
                         "--transcriber", "cjm-capability-voxtral-hf",
                         "--graph-capability", "cjm-capability-graph-sqlite",
                         "--graph-db-path", "/tmp/g.db"])
    assert args.transcriber == ["cjm-capability-whisper", "cjm-capability-voxtral-hf"]
    assert args.graph_capability == "cjm-capability-graph-sqlite"
    assert args.graph_db_path == "/tmp/g.db"


def test_sysmon_and_preprocessing_flags():
    p = build_parser()
    args = p.parse_args(["run", "a.mp3", "--sysmon-capability", "cjm-capability-monitor-nvidia"])
    assert args.sysmon_capability == "cjm-capability-monitor-nvidia"
    # opt-in preprocessing (stage 8: Demucs vocals isolation before the model-input convert)
    args = p.parse_args(["run", "a.mp3", "--preprocessing-capability", "cjm-capability-demucs"])
    assert args.preprocessing_capability == "cjm-capability-demucs"


def test_max_concurrent_parsing():
    # repeatable --max-concurrent (stage-5 closeout: SG-33 per-capability cap override)
    p = build_parser()
    args = p.parse_args(["run", "a.mp3",
                         "--max-concurrent", "cjm-capability-ffmpeg=4",
                         "--max-concurrent", "cjm-capability-silero-vad=2"])
    assert parse_max_concurrent(args.max_concurrent) == {
        "cjm-capability-ffmpeg": 4, "cjm-capability-silero-vad": 2}
    assert parse_max_concurrent(None) == {}


@pytest.mark.parametrize("bad", ["nocap", "x=", "x=zero", "x=0"])
def test_max_concurrent_refuses_loudly(bad):
    # malformed / non-int / below-1 all refuse loudly
    with pytest.raises(SystemExit):
        parse_max_concurrent([bad])


def test_transcriber_spec_parsing():
    # bare name = default instance (stage-5 behavior, backward compatible)
    assert parse_transcriber_spec("cjm-capability-whisper") == {
        "capability": "cjm-capability-whisper",
        "instance_id": "cjm-capability-whisper", "config": {}}
    # @INSTANCE_ID + config overrides = CR-10 named (capability, MODEL) instance
    assert parse_transcriber_spec("cjm-capability-whisper@whisper-tiny:model=tiny") == {
        "capability": "cjm-capability-whisper",
        "instance_id": "whisper-tiny", "config": {"model": "tiny"}}
    # value coercion: string (incl. a '/'-bearing HF model id) / bool / float / int
    assert parse_transcriber_spec(
        "cjm-capability-voxtral-hf@voxtral-small:model_id=mistralai/Voxtral-Small-24B-2507,"
        "do_sample=true,temperature=0.7,max_new_tokens=100")["config"] == {
        "model_id": "mistralai/Voxtral-Small-24B-2507", "do_sample": True,
        "temperature": 0.7, "max_new_tokens": 100}


@pytest.mark.parametrize("bad", [
    "", "@x", "name@",            # empty / nameless / dangling '@'
    "name:model=tiny",            # config override without an addressable @INSTANCE_ID
    "name@i:model", "name@i:=v", "name@i:k=",  # malformed key=value pairs
])
def test_transcriber_spec_refuses_loudly(bad):
    with pytest.raises(SystemExit):
        parse_transcriber_spec(bad)


def test_expand_sources_files_and_folders(tmp_path):
    d = tmp_path / "feed"
    (d / "sub").mkdir(parents=True)
    (d / "b.mp3").write_bytes(b"x")
    (d / "sub" / "a.wav").write_bytes(b"x")
    (d / "notes.txt").write_bytes(b"x")  # non-media: skipped by directory expansion
    lone = tmp_path / "lone.opus"
    lone.write_bytes(b"x")
    out = expand_sources([str(lone), str(d)])
    # explicit files pass through first (caller order), then the folder's media
    # files in sorted-path order (deterministic feedstock runs)
    assert out == [str(lone.resolve()),
                   str((d / "b.mp3").resolve()),
                   str((d / "sub" / "a.wav").resolve())]


def test_expand_sources_refuses_loudly(tmp_path):
    # a missing explicit file refuses (same contract the inline check had)
    with pytest.raises(SystemExit):
        expand_sources([str(tmp_path / "missing.mp3")])
    # a directory with no media files refuses instead of silently running nothing
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit):
        expand_sources([str(empty)])


def test_expand_sources_with_collections(tmp_path):
    d = tmp_path / "Hardcore_History"
    d.mkdir()
    (d / "ep1.mp3").write_bytes(b"x")
    (d / "ep2.mp3").write_bytes(b"x")
    lone = tmp_path / "lone.opus"
    lone.write_bytes(b"x")

    # folder gesture -> ONE proposed, ordered declaration; bare files ride outside it
    files, decls = expand_sources_with_collections([str(lone), str(d)])
    assert files == [str(lone.resolve()),
                     str((d / "ep1.mp3").resolve()),
                     str((d / "ep2.mp3").resolve())]
    assert len(decls) == 1
    decl = decls[0]
    assert decl.title == "Hardcore History", "folder name prettified for display"
    assert decl.status == "proposed" and decl.ordered is True
    assert decl.member_paths == files[1:], "only the folder's expansion files in"

    # --collection TITLE: one human-named CONFIRMED declaration over ALL sources;
    # a hand-typed file list carries no fabricated order
    files2, decls2 = expand_sources_with_collections(
        [str(lone), str(d)], explicit_title="My Book", actor="cli:cj")
    assert len(decls2) == 1
    assert decls2[0].status == "confirmed" and decls2[0].actor == "cli:cj"
    assert decls2[0].member_paths == files2 and decls2[0].ordered is False

    # single-folder run with an explicit title keeps the expansion order
    _, decls3 = expand_sources_with_collections([str(d)], explicit_title="My Book")
    assert decls3[0].ordered is True

    # --no-collection expands only; the flag pair refuses loudly
    _, none_decls = expand_sources_with_collections([str(d)], no_collection=True)
    assert none_decls == []
    with pytest.raises(SystemExit):
        expand_sources_with_collections([str(d)], explicit_title="X", no_collection=True)
