"""Tests for cjm_transcription_core.cli — parser smoke checks (no capabilities involved).

Projected from the cli notebook's parser-check cell at the golden-reference flip."""
import pytest

from cjm_transcription_core.cli import build_parser, parse_max_concurrent


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
