"""Tests for the TUI's pure logic — browser/selection state, candidate
enumeration off fake manifests, spec round-trips through the CORE parser,
and plan argv (no Textual, no capabilities involved)."""

import json
import re
from pathlib import Path

from cjm_transcription_core.cli import build_parser as core_build_parser
from cjm_transcription_core.cli import parse_transcriber_spec
from cjm_transcription_core.candidates import (candidate_directives, discover_capability,
                                              instance_id_for, model_axis, spec_string,
                                              transcription_manifests)
from cjm_transcription_core.launch import build_parser, plan_argv
from cjm_transcription_core.results import RunIndex
from cjm_transcription_core.sources import CollectionField, SourceBrowser
from cjm_transcription_core.state import load_state, save_state, state_path


def _write_manifest(d, name, methods, props):
    """Minimal capability-manifest json with just the fields enumeration reads."""
    (d / f"{name}.json").write_text(json.dumps({
        "code": {"name": name,
                 "structural_surface": {"methods": [{"name": m} for m in methods]},
                 "config_schema": {"properties": props}}}))


def test_candidate_enumeration(tmp_path):
    _write_manifest(tmp_path, "cjm-capability-whisper", ["transcribe", "cancel"],
                    {"model": {"enum": ["tiny", "base"], "default": "base"}})
    _write_manifest(tmp_path, "cjm-capability-voxtral-hf", ["transcribe"],
                    {"model_id": {"enum": ["org/Mini-3B", "org/Small-24B"],
                                  "default": "org/Mini-3B"}})
    # surface-matched: a media tool (no transcribe method) and an adapter unit
    # (no code section) must both stay out of the candidate space
    _write_manifest(tmp_path, "cjm-capability-ffmpeg", ["convert", "segment_audio"], {})
    (tmp_path / "adapter-transcription-x.json").write_text(json.dumps({"unit": "adapter"}))

    assert set(transcription_manifests(str(tmp_path))) == {
        "cjm-capability-whisper", "cjm-capability-voxtral-hf"}
    rows = candidate_directives(str(tmp_path))
    assert len(rows) == 4  # 2 whisper models + 2 voxtral models
    # the DEFAULT model keeps the bare capability name (stage-5/cache continuity)
    assert {r["instance_id"] for r in rows if r["default"]} == {
        "cjm-capability-whisper", "cjm-capability-voxtral-hf"}
    assert {r["instance_id"] for r in rows if not r["default"]} == {
        "whisper--tiny", "voxtral-hf--Small-24B"}
    # the model axis key is data-derived per capability, not hard-coded
    named = {r["instance_id"]: r for r in rows if not r["default"]}
    assert named["whisper--tiny"]["config"] == {"model": "tiny"}
    assert named["voxtral-hf--Small-24B"]["config"] == {"model_id": "org/Small-24B"}


def test_spec_round_trip_and_instance_ids(tmp_path):
    _write_manifest(tmp_path, "cjm-capability-whisper", ["transcribe"],
                    {"model": {"enum": ["tiny", "large-v3"], "default": "tiny"}})
    _write_manifest(tmp_path, "cjm-capability-voxtral-hf", ["transcribe"],
                    {"model_id": {"enum": ["mistralai/Voxtral-Small-24B-2507"],
                                  "default": None}})
    for r in candidate_directives(str(tmp_path)):
        # every directive the TUI can produce must survive the CORE parser
        # unchanged — the hand-off contract with cjm-transcription-core run
        assert parse_transcriber_spec(spec_string(r)) == {
            "capability": r["capability"], "instance_id": r["instance_id"],
            "config": r["config"]}
        # and every derived id must satisfy the manager's CR-10 pattern
        assert re.fullmatch(r"[A-Za-z0-9_-]{1,64}", r["instance_id"])
    assert instance_id_for("cjm-capability-voxtral-hf",
                           "mistralai/Voxtral-Small-24B-2507") == \
        "voxtral-hf--Voxtral-Small-24B-2507"
    assert model_axis({"config_schema": {"properties": {"device": {"enum": ["cpu"]}}}}) is None


def test_source_browser_walk_and_selection(tmp_path):
    (tmp_path / "pod").mkdir()
    (tmp_path / "pod" / "ep1.mp3").write_bytes(b"x")
    (tmp_path / "b.wav").write_bytes(b"x")
    (tmp_path / "notes.txt").write_bytes(b"x")   # non-media: not browsable
    (tmp_path / ".hidden").mkdir()               # hidden dir: not browsable

    b = SourceBrowser(str(tmp_path))
    assert [e.name for e in b.entries()] == ["pod", "b.wav"]
    b.enter()                                    # descend pod/
    assert b.cwd == (tmp_path / "pod").resolve()
    b.enter()                                    # toggle-select ep1.mp3
    assert b.selected == [str((tmp_path / "pod" / "ep1.mp3").resolve())]
    b.up()
    b.move(1)                                    # focus b.wav
    b.enter()                                    # select it (order = pick order)
    assert [Path(s).name for s in b.selected] == ["ep1.mp3", "b.wav"]
    b.enter()                                    # toggle again = deselect
    assert [Path(s).name for s in b.selected] == ["ep1.mp3"]
    b.move(-1)                                   # focus pod/ again
    b.add_folder()                               # folder-source, kept unexpanded
    assert b.selected[-1] == str((tmp_path / "pod").resolve())


def test_plan_argv_parses_in_the_core_cli(tmp_path):
    # graph/sysmon ride the PLAN (resolved flags > state > discovery in main);
    # only preprocessing/actor still pass through from the raw args
    args = build_parser().parse_args(["--actor", "tui:test"])
    src = tmp_path / "ep1.mp3"
    src.write_bytes(b"x")
    plan = {"sources": [str(src)],
            "transcribers": ["cjm-capability-whisper@whisper--tiny:model=tiny",
                             "cjm-capability-voxtral-hf"],
            "lightweight": "whisper--tiny", "accuracy": "cjm-capability-voxtral-hf",
            "max_segment_duration": 220.0, "sysmon_capability": None,
            "graph_capability": "cjm-capability-graph-sqlite",
            "graph_db_path": "/tmp/g.db",
            "manifests_dir": ".cjm/manifests"}
    argv = plan_argv(plan, args)
    # the hand-off contract: the rendered argv must parse in the CORE CLI
    parsed = core_build_parser().parse_args(argv)
    assert parsed.command == "run"
    assert parsed.audio == [str(src)]
    assert parsed.yes is True
    assert parsed.transcriber == plan["transcribers"]
    assert parsed.graph_capability == "cjm-capability-graph-sqlite"
    assert parsed.graph_db_path == "/tmp/g.db"
    assert parsed.actor == "tui:test"
    assert parsed.sysmon_capability is None
    # no toggle judgment, no flag -> preprocessing stays off in the hand-off
    assert parsed.preprocessing_capability is None
    # a plan WITHOUT the diarization key (scripted caller) renders NEITHER
    # flag — the core's own default applies
    assert parsed.diarization_capability is None
    assert parsed.no_diarization is False

    # 450e7c78: a TUI plan always carries the key, and the argv renders the
    # choice EXPLICITLY both ways (replay-stable against core default drift)
    plan["diarization_capability"] = "cjm-capability-pyannote"
    parsed = core_build_parser().parse_args(plan_argv(plan, args))
    assert parsed.diarization_capability == "cjm-capability-pyannote"
    assert parsed.no_diarization is False
    plan["diarization_capability"] = None
    parsed = core_build_parser().parse_args(plan_argv(plan, args))
    assert parsed.diarization_capability is None
    assert parsed.no_diarization is True
    del plan["diarization_capability"]

    # the in-TUI A/B verdict rides the plan and WINS over the passthrough flag;
    # toggled-off (plan None) falls back to the flag for scripted runs
    plan["preprocessing_capability"] = "cjm-capability-demucs"
    parsed = core_build_parser().parse_args(plan_argv(plan, args))
    assert parsed.preprocessing_capability == "cjm-capability-demucs"
    flag_args = build_parser().parse_args(
        ["--preprocessing-capability", "cjm-capability-other"])
    parsed = core_build_parser().parse_args(plan_argv(plan, flag_args))
    assert parsed.preprocessing_capability == "cjm-capability-demucs"
    plan["preprocessing_capability"] = None
    parsed = core_build_parser().parse_args(plan_argv(plan, flag_args))
    assert parsed.preprocessing_capability == "cjm-capability-other"


def test_state_roundtrip_and_role_discovery(tmp_path):
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    # role discovery is surface-based: graph storage = add_nodes, monitor =
    # get_system_status; a transcriber must match NEITHER role
    _write_manifest(manifests, "cjm-capability-graph-sqlite",
                    ["add_nodes", "add_edges", "find_nodes_by_label"], {})
    _write_manifest(manifests, "cjm-capability-monitor-nvidia",
                    ["get_system_status", "list_processes"], {})
    _write_manifest(manifests, "cjm-capability-whisper", ["transcribe"], {})
    _write_manifest(manifests, "cjm-capability-demucs",
                    ["separate_vocals", "separate_stems"], {})
    _write_manifest(manifests, "cjm-capability-pyannote", ["diarize"], {})
    assert discover_capability(str(manifests), "add_nodes") == "cjm-capability-graph-sqlite"
    assert discover_capability(str(manifests), "get_system_status") == "cjm-capability-monitor-nvidia"
    # preprocessing A/B (5aba2ab6): the d-toggle capability is role-discovered too
    assert discover_capability(str(manifests), "separate_vocals") == "cjm-capability-demucs"
    # diarization rung (450e7c78): the s-toggle capability is role-discovered too
    assert discover_capability(str(manifests), "diarize") == "cjm-capability-pyannote"
    assert discover_capability(str(manifests), "no_such_method") is None

    # state sidecar lands NEXT TO the manifests dir and round-trips merges
    assert load_state(str(manifests)) == {}
    save_state(str(manifests), graph_capability="cjm-capability-graph-sqlite",
               picked_instance_ids=["whisper--tiny"])
    save_state(str(manifests), last_cwd="/data/podcasts")
    state = load_state(str(manifests))
    assert state == {"graph_capability": "cjm-capability-graph-sqlite",
                     "picked_instance_ids": ["whisper--tiny"],
                     "last_cwd": "/data/podcasts"}
    assert state_path(str(manifests)).parent == tmp_path


def test_source_browser_listing_cache(tmp_path):
    # Per-event navigation must NOT re-enumerate the cwd (finding 3a3db22c:
    # a full iterdir+stat pass per free-spin wheel tick froze the UI)
    (tmp_path / "a.wav").write_bytes(b"x")
    b = SourceBrowser(str(tmp_path))
    rows = b.entries()
    assert [e.name for e in rows] == ["a.wav"]
    assert b.entry_keys() == [str((tmp_path / "a.wav").resolve())]
    # a file added behind the cache stays invisible to plain navigation...
    (tmp_path / "b.wav").write_bytes(b"x")
    assert b.entries() is rows                  # same cached object, no rescan
    # ...until an explicit refresh (the app calls it on stage re-entry)
    b.refresh()
    assert [e.name for e in b.entries()] == ["a.wav", "b.wav"]
    # cwd changes invalidate naturally (enter/up)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.mp3").write_bytes(b"x")
    b.refresh()
    b.enter()                                   # descend sub/ (dirs sort first)
    assert b.cwd == (tmp_path / "sub").resolve()
    assert [e.name for e in b.entries()] == ["c.mp3"]
    b.up()
    assert [e.name for e in b.entries()][0] == "sub"
    # keys stay row-aligned with entries after every re-enumeration
    assert len(b.entry_keys()) == len(b.entries())


def test_run_index(tmp_path):
    # RunIndex (6768bafb): tolerant load + newest-first order, path-keyed
    # counts (browse chips), dir prefix counts, content-identity hash-check
    from cjm_substrate.utils.hashing import hash_file
    from cjm_transcription_core.results import RunIndex

    media = tmp_path / "library"
    media.mkdir()
    src = media / "ep1.mp3"
    src.write_bytes(b"the-audio-bytes")
    digest = hash_file(str(src))

    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "run_a.json").write_text(json.dumps({
        "format": "cjm-transcription-core/run-manifest",
        "run_id": "run_a", "created_at": 100.0,
        "config": {"transcriber_capabilities": ["cjm-capability-whisper"]},
        "sources": [{"source_path": str(src), "content_hash": digest,
                     "segments": [{"index": 0, "start": 0.0, "end": 1.0,
                                   "transcripts": {"cjm-capability-whisper":
                                                   {"text": "hello"}}}]}]}))
    # newer run references the SAME CONTENT under a different (moved) path
    (runs / "run_b.json").write_text(json.dumps({
        "format": "cjm-transcription-core/run-manifest",
        "run_id": "run_b", "created_at": 200.0,
        "config": {"transcriber_capabilities": ["cjm-capability-voxtral-hf"]},
        "sources": [{"source_path": str(tmp_path / "elsewhere.mp3"),
                     "content_hash": digest,
                     "segments": [{"index": 0, "text": "old flat schema"}]}]}))
    (runs / "not-a-manifest.json").write_text("{broken")   # skipped, never raises
    (runs / "foreign.json").write_text(json.dumps({"x": 1}))  # foreign json skipped
    # decomp manifests share the runs/ dir and MATCH the run_id+sources shape —
    # only the format tag keeps them out of the transcription list (5329fbd8)
    (runs / "decomp_z.json").write_text(json.dumps({
        "format": "cjm-transcript-decomp-core/run-manifest",
        "run_id": "decomp_z", "created_at": 300.0, "sources": []}))

    idx = RunIndex(str(runs))
    assert idx.load() == 2
    assert [m["run_id"] for m in idx.runs] == ["run_b", "run_a"]  # newest first
    assert idx.transcribers(idx.runs[0]) == ["cjm-capability-voxtral-hf"]

    counts = idx.counts_by_path()
    assert counts[str(src.resolve())] == 1
    assert RunIndex.dir_count(counts, str(media.resolve())) == 1
    assert RunIndex.dir_count(counts, str(tmp_path.resolve())) == 2

    # content identity: BOTH runs match the file, including the moved-path one
    res = idx.hash_check(str(src))
    assert res["content_hash"] == digest
    assert [h["run_id"] for h in res["matches"]] == ["run_b", "run_a"]


def test_collection_field_and_plan_argv_mapping(tmp_path):
    d = tmp_path / "Hardcore_History"
    d.mkdir()
    (d / "ep1.mp3").write_bytes(b"x")
    lone = tmp_path / "lone.mp3"
    lone.write_bytes(b"x")

    # auto: proposals preview exactly what an untouched run would propose
    f = CollectionField()
    sel = [str(lone), str(d)]
    assert f.proposals(sel) == ["Hardcore History"]
    assert f.prefill(sel) == "Hardcore History", "single folder seeds the editor"
    text, mode = f.summary(sel)
    assert mode == "auto" and "Hardcore History" in text

    # named: the operator touched it; empty submit falls back to auto
    f.set_named("  Supernova in the East ")
    assert (f.mode, f.title) == ("named", "Supernova in the East")
    assert f.prefill(sel) == "Supernova in the East"
    f.set_named("")
    assert (f.mode, f.title) == ("auto", None)

    # off: declining is a toggle, never sticky state
    f.toggle_off()
    assert f.summary(sel) == ("none  (--no-collection)", "off")
    f.toggle_off()
    assert f.mode == "auto"

    # plan_argv contract: each mode parses in the CORE CLI as the right flags
    args = build_parser().parse_args([])
    plan = {"sources": [str(lone)], "transcribers": ["cjm-capability-whisper"],
            "lightweight": "w", "accuracy": "w", "max_segment_duration": 220.0,
            "sysmon_capability": None, "graph_capability": None,
            "graph_db_path": None, "manifests_dir": ".cjm/manifests"}
    parsed = core_build_parser().parse_args(plan_argv(plan, args))
    assert parsed.collection is None and parsed.no_collection is False, \
        "no plan field (or auto) passes nothing — the core proposes"
    plan["collection"] = {"mode": "named", "title": "Supernova in the East"}
    parsed = core_build_parser().parse_args(plan_argv(plan, args))
    assert parsed.collection == "Supernova in the East"
    plan["collection"] = {"mode": "off", "title": None}
    parsed = core_build_parser().parse_args(plan_argv(plan, args))
    assert parsed.no_collection is True

    # pick-existing surface: recent titles out of past manifests, newest first
    runs = tmp_path / "runs"
    runs.mkdir()
    for i, title in enumerate(["Old Book", "Hardcore History"]):
        (runs / f"r{i}.json").write_text(json.dumps({
            "format": "cjm-transcription-core/run-manifest", "run_id": f"r{i}",
            "created_at": float(i), "sources": [],
            "collections": [{"title": title}]}))
    idx = RunIndex(str(runs))
    idx.load()
    assert idx.collection_titles() == ["Hardcore History", "Old Book"]
