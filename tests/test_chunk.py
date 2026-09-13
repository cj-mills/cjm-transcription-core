"""Tests for cjm_transcription_core.chunk — chunk-grain landings, derived manifests, the census
(pure; no capabilities involved). Work item cf0b91d6, rulings 9ffce5f7 · 8a9b9639 · 910f3692."""
import asyncio
import json

import pytest

from cjm_transcription_core import chunk
from cjm_transcription_core.chunk import (apply_chunk_update, build_chunk_landing, census_rows,
                                           chunks_from_census, derive_manifest, land_chunk_transcript,
                                           load_run_manifest, prior_config_hash, prompt_hash_of,
                                           save_manifest, select_chunks, summarize_census)
from cjm_transcript_graph_schema.schema import (audio_rendition_node_id, audio_segment_node_id,
                                                external_config_hash, external_transcriber_name,
                                                source_node_id, transcript_node_id)


def _manifest():
    return {
        "format": "cjm-transcription-core/run-manifest", "version": "0.5.0",
        "run_id": "run_parent", "created_at": 1.0,
        "config": {"transcriber_capabilities": ["whisper--small", "voxtral"]},
        "capabilities": {"whisper--small": {"name": "cjm-capability-whisper", "config_hash": "sha256:cw"},
                         "voxtral": {"name": "cjm-capability-voxtral-hf", "config_hash": "sha256:cv-old"}},
        "sources": [
            {"source_path": "/media/Lecture 33: Bitblas.mp4", "content_hash": "sha256:src33", "chain": [],
             "segments": [
                 {"index": 0, "start": 0.0, "end": 220.0, "model_input_path": "/ws/s0.wav", "model_input_hash": "sha256:w0",
                  "transcripts": {"whisper--small": {"job_id": "a", "text": "fine", "metadata": {}},
                                  "voxtral": {"job_id": "b", "text": "fine too", "metadata": {}}}},
                 {"index": 1, "start": 220.0, "end": 440.0, "model_input_path": "/ws/s1.wav", "model_input_hash": "sha256:w1",
                  "transcripts": {"whisper--small": {"job_id": "c", "text": "ok", "metadata": {}},
                                  "voxtral": {"job_id": "d", "text": "some " * 25000, "metadata": {}}}},
             ]},
            {"source_path": "/media/Lecture 17: NCCL.mp4", "content_hash": "sha256:src17", "chain": ["source_separation:demucs@x"],
             "segments": [
                 {"index": 0, "start": 0.0, "end": 200.0, "model_input_path": "/ws/t0.wav", "model_input_hash": "sha256:v0",
                  "transcripts": {"whisper--small": {"job_id": "e", "text": "nickel", "metadata": {}},
                                  "voxtral": {"job_id": "f", "text": "nickel", "metadata": {}}}},
             ]},
        ],
        "graph": {"capability": "cjm-capability-graph-sqlite", "db_path": "/ws/.cjm/g.db"},
        "collections": [],
    }


def test_select_chunks_by_index_hash_path_and_refusals():
    m = _manifest()
    assert [(si, seg["index"]) for si, _, seg in select_chunks(m)] == [(0, 0), (0, 1), (1, 0)]
    assert [(si, seg["index"]) for si, _, seg in select_chunks(m, source="1")] == [(1, 0)]
    assert [(si, seg["index"]) for si, _, seg in select_chunks(m, source="src33", segments=[1])] == [(0, 1)]
    assert [(si, seg["index"]) for si, _, seg in select_chunks(m, source="sha256:src17")] == [(1, 0)]
    assert [(si, seg["index"]) for si, _, seg in select_chunks(m, source="NCCL")] == [(1, 0)]
    with pytest.raises(ValueError):
        select_chunks(m, source="Lecture")  # two matches
    with pytest.raises(ValueError):
        select_chunks(m, source="7")
    with pytest.raises(ValueError):
        select_chunks(m, source="0", segments=[5])


def test_prior_config_hash_prefers_the_entry_override():
    m = _manifest()
    seg = m["sources"][0]["segments"][1]
    assert prior_config_hash(m, seg, "voxtral") == "sha256:cv-old"
    seg["transcripts"]["voxtral"]["config_hash"] = "sha256:cv-mid"
    assert prior_config_hash(m, seg, "voxtral") == "sha256:cv-mid"
    assert prior_config_hash(m, seg, "gemini-2.5-pro/manual") == ""


def test_build_chunk_landing_recomputes_ids_and_supersedes_the_prior_variant():
    m = _manifest()
    src, seg = m["sources"][0], m["sources"][0]["segments"][1]
    nodes, edges, ids = build_chunk_landing(src, seg, transcriber="voxtral", config_hash="sha256:cv-new",
                                            text="the good prefix some", metadata={"degenerate_tail": {"repeats": 9}},
                                            prior_config_hash="sha256:cv-old")
    sid = source_node_id("sha256:src33")
    aseg = audio_segment_node_id(sid, 220.0, 440.0)
    rend = audio_rendition_node_id(aseg, [])
    assert ids == {"source": sid, "audio_segment": aseg, "rendition": rend,
                   "transcript": transcript_node_id(rend, "voxtral", "sha256:cv-new"),
                   "supersedes": transcript_node_id(rend, "voxtral", "sha256:cv-old")}
    [node] = nodes
    assert node["label"] == "Transcript" and node["properties"]["text"] == "the good prefix some"
    assert node["properties"]["metadata"]["degenerate_tail"] == {"repeats": 9}
    assert node["properties"]["actor"] == "capability:voxtral"
    assert [(e["source_id"], e["target_id"], e["relation_type"]) for e in edges] == [
        (ids["transcript"], rend, "DERIVED_FROM"), (ids["transcript"], ids["supersedes"], "SUPERSEDES")]
    # The chain rides into the rendition id (a demucs source lands under the vocals rendition).
    src2, seg2 = m["sources"][1], m["sources"][1]["segments"][0]
    _, _, ids2 = build_chunk_landing(src2, seg2, transcriber="voxtral", config_hash="sha256:cv-new", text="NCCL")
    assert ids2["rendition"] == audio_rendition_node_id(audio_segment_node_id(source_node_id("sha256:src17"), 0.0, 200.0),
                                                        ["source_separation:demucs@x"])
    # No prior / same hash -> no SUPERSEDES edge.
    _, edges3, ids3 = build_chunk_landing(src, seg, transcriber="voxtral", config_hash="sha256:cv-old", text="x",
                                          prior_config_hash="sha256:cv-old")
    assert ids3["supersedes"] is None and [e["relation_type"] for e in edges3] == ["DERIVED_FROM"]
    # External landing: operator attribution + the external identity helpers.
    name = external_transcriber_name("gemini-2.5-pro")
    h = external_config_hash("gemini-2.5-pro", prompt_hash_of("Transcribe the lecture chunk."))
    nodes4, _, ids4 = build_chunk_landing(src2, seg2, transcriber=name, config_hash=h, text="NCCL",
                                          actor="human:cj", method="external-landing")
    assert ids4["transcript"] == transcript_node_id(ids2["rendition"], name, h)
    assert nodes4[0]["properties"]["actor"] == "human:cj" and nodes4[0]["properties"]["method"] == "external-landing"
    # Identity guards fire loudly.
    with pytest.raises(ValueError):
        build_chunk_landing({**src, "content_hash": ""}, seg, transcriber="voxtral", config_hash="h", text="x")
    with pytest.raises(ValueError):
        build_chunk_landing(src, {**seg, "model_input_hash": ""}, transcriber="voxtral", config_hash="h", text="x")
    with pytest.raises(ValueError):
        build_chunk_landing(src, seg, transcriber="voxtral", config_hash="", text="x")


def test_land_chunk_transcript_journals_the_delta_with_the_landing_verb(tmp_path, monkeypatch):
    """The ONE landing: journal_extend with verb `transcript-landing`, the producer + reason +
    ids in args, run pinned — both producers share it. The extend is recorded, not run."""
    calls = []

    class _Res:
        nodes_added, nodes_verified, edges_added, edges_existing = 1, 0, 2, 0
        added_node_ids = added_edge_ids = ()

    async def fake_extend(queue, graph_id, nodes, edges, **kw):
        calls.append((graph_id, nodes, edges, kw))
        return _Res()
    monkeypatch.setattr(chunk, "journal_extend", fake_extend)
    m = _manifest()
    src, seg = m["sources"][0], m["sources"][0]["segments"][1]
    rec = asyncio.run(land_chunk_transcript(
        None, "g", src, seg, transcriber="voxtral", config_hash="sha256:cv-new", text="prefix",
        metadata={"landing": {"producer": "rerun"}}, prior_config_hash="sha256:cv-old",
        producer="rerun", reason="runaway", actor="cli:cj", run_id="run_rerun",
        journal_path=str(tmp_path / "g.writes.jsonl")))
    [(gid, nodes, edges, kw)] = calls
    assert gid == "g" and len(nodes) == 1 and len(edges) == 2
    assert kw["verb"] == "transcript-landing" and kw["actor"] == "cli:cj" and kw["run"] == "run_rerun"
    assert kw["journal_path"].endswith("g.writes.jsonl")
    assert kw["args"]["producer"] == "rerun" and kw["args"]["reason"] == "runaway"
    assert kw["args"]["transcript_id"] == rec["transcript"] and kw["args"]["supersedes"] == rec["supersedes"]
    assert kw["args"]["segment_index"] == 1 and kw["args"]["transcriber"] == "voxtral"
    assert rec["nodes_added"] == 1 and rec["edges_added"] == 2


def test_derived_manifest_replaces_only_the_touched_entry_and_registers_a_new_transcriber(tmp_path):
    parent = _manifest()
    parent_json = json.dumps(parent)
    d = derive_manifest(parent, run_id="run_rerun", parent_path="/ws/runs/run_parent.json", kind="rerun-chunk", created_at=2.0)
    assert d["run_id"] == "run_rerun" and d["parent_run_id"] == "run_parent"
    assert d["parent_manifest"] == "/ws/runs/run_parent.json" and d["derivation"] == {"kind": "rerun-chunk", "landings": []}
    assert d["format"] == parent["format"], "the derived manifest stays a transcription-core run manifest"
    row = apply_chunk_update(d, 0, 1, "voxtral", {"job_id": "j", "text": "prefix", "metadata": {},
                                                  "config_hash": "sha256:cv-new",
                                                  "landing": {"producer": "rerun", "reason": "runaway", "supersedes": "t-old"}})
    assert row == {"source_index": 0, "segment_index": 1, "transcriber": "voxtral", "config_hash": "sha256:cv-new",
                   "producer": "rerun", "reason": "runaway", "supersedes": "t-old"}
    assert d["sources"][0]["segments"][1]["transcripts"]["voxtral"]["text"] == "prefix"
    assert d["sources"][0]["segments"][1]["transcripts"]["voxtral"]["config_hash"] == "sha256:cv-new"
    assert d["sources"][0]["segments"][0]["transcripts"]["voxtral"]["text"] == "fine too", "untouched entry kept"
    assert d["capabilities"]["voxtral"]["config_hash"] == "sha256:cv-old", "run-level block untouched by a re-run"
    assert d["derivation"]["landings"] == [row]
    assert json.dumps(parent) == parent_json, "the parent is never mutated"
    # An external landing registers the new transcriber + its capabilities record.
    name = external_transcriber_name("gemini-2.5-pro")
    apply_chunk_update(d, 1, 0, name, {"job_id": "x", "text": "NCCL", "metadata": {}, "config_hash": "sha256:ext",
                                       "landing": {"producer": "external"}},
                       capability_info={"name": name, "version": "manual", "config_hash": "sha256:ext",
                                        "config": {"model_id": "gemini-2.5-pro"}})
    assert d["config"]["transcriber_capabilities"] == ["whisper--small", "voxtral", name]
    assert d["capabilities"][name]["version"] == "manual"
    assert name not in d["sources"][0]["segments"][0]["transcripts"], "present only where it landed"
    with pytest.raises(ValueError):
        apply_chunk_update(d, 0, 9, "voxtral", {"config_hash": "h"})
    with pytest.raises(ValueError):
        apply_chunk_update(d, 0, 0, "voxtral", {"text": "no hash"})
    # Round-trip through save + load keeps the format and the derivation header.
    out = save_manifest(d, tmp_path / "runs" / "run_rerun.json")
    back = load_run_manifest(out)
    assert back["parent_run_id"] == "run_parent" and back["derivation"]["kind"] == "rerun-chunk"
    assert back["sources"][0]["segments"][1]["transcripts"]["voxtral"]["config_hash"] == "sha256:cv-new"
    (tmp_path / "foreign.json").write_text(json.dumps({"format": "cjm-transcript-decomp-core/run-manifest", "sources": []}))
    with pytest.raises(ValueError):
        load_run_manifest(tmp_path / "foreign.json")


def _rows():
    def row(coll, src, idx, transcriber, chars, words, *, degenerate=0, superseded=0, start=0.0, end=220.0):
        return {"collection": coll, "source_id": f"S-{src}", "source_path": f"/m/{src}.mp4", "content_hash": f"sha256:{src}",
                "audio_segment": f"A-{src}-{idx}", "seg_index": idx, "start": start, "end": end,
                "rendition_id": f"R-{src}-{idx}", "transcriber": transcriber, "config_hash": "h",
                "transcript_id": f"T-{src}-{idx}-{transcriber}", "chars": chars, "words": words,
                "degenerate": degenerate, "producer": None, "superseded": superseded}
    return [
        row("GPU MODE", "l33", 0, "whisper--small", 2000, 400), row("GPU MODE", "l33", 0, "voxtral", 2100, 410),
        row("GPU MODE", "l33", 1, "whisper--small", 2154, 420), row("GPU MODE", "l33", 1, "voxtral", 124835, 24965),  # runaway
        row("GPU MODE", "l33", 2, "whisper--small", 1900, 380), row("GPU MODE", "l33", 2, "voxtral", 900, 169, degenerate=1),  # guarded
        row("GPU MODE", "l33", 3, "whisper--small", 100, 25), row("GPU MODE", "l33", 3, "voxtral", 30, 4),  # silence-ish: below min_words? hi=25 >= 20, lo=4 -> ratio 6.25
        row("GPU MODE_OLD", "old", 0, "voxtral", 30000, 6000, superseded=1),  # superseded: not live
        row("Dumbing Us Down", "dud", 0, "whisper--small", 2000, 400), row("Dumbing Us Down", "dud", 0, "voxtral", 2050, 405),
    ]


def test_census_flags_runaways_degenerate_and_disagreement_and_excludes_superseded():
    rows = _rows()
    flagged = census_rows(rows)
    by_id = {f["transcript_id"]: f for f in flagged}
    assert set(by_id) == {"T-l33-1-voxtral", "T-l33-1-whisper--small", "T-l33-2-voxtral", "T-l33-3-voxtral", "T-l33-3-whisper--small"}
    assert by_id["T-l33-1-voxtral"]["reasons"] == ["oversized", "implausible_rate", "disagreement"]
    assert by_id["T-l33-1-voxtral"]["words_per_second"] == pytest.approx(24965 / 220, abs=0.01)
    assert by_id["T-l33-1-whisper--small"]["reasons"] == ["disagreement"], "the peer is listed with the ratio"
    assert by_id["T-l33-1-whisper--small"]["disagreement_ratio"] == pytest.approx(24965 / 420, abs=0.1)
    assert by_id["T-l33-2-voxtral"]["reasons"] == ["degenerate"]
    assert by_id["T-l33-3-voxtral"]["reasons"] == ["disagreement"]
    # Superseded variants are not live; asked for, they appear marked.
    assert "T-old-0-voxtral" not in by_id
    assert "T-old-0-voxtral" in {f["transcript_id"] for f in census_rows(rows, include_superseded=True)}
    # Restrictions: one transcriber; one collection; thresholds.
    assert {f["transcriber"] for f in census_rows(rows, transcriber="voxtral")} == {"voxtral"}
    assert census_rows(rows, collections=["Dumbing Us Down"]) == []
    assert [f["transcript_id"] for f in census_rows(rows, collections=["GPU MODE"], transcriber="voxtral", disagreement_ratio=1000)] == [
        "T-l33-1-voxtral", "T-l33-2-voxtral"]
    assert "oversized" not in census_rows(rows, max_chars=200000, transcriber="voxtral")[0]["reasons"]
    # The roll-up: live totals + flagged by reason per collection.
    summary = summarize_census(flagged, rows)
    assert summary["GPU MODE"]["live"] == 8 and summary["GPU MODE"]["flagged"] == 5
    assert summary["GPU MODE"]["by_reason"]["oversized"] == 1 and summary["GPU MODE"]["by_transcriber"]["voxtral"] == 3
    assert summary["GPU MODE_OLD"] == {"flagged": 0, "live": 0, "by_reason": {}, "by_transcriber": {}}
    assert summary["Dumbing Us Down"]["flagged"] == 0 and summary["Dumbing Us Down"]["live"] == 2


def test_chunks_from_census_maps_rows_onto_manifest_chunks():
    m = _manifest()
    sid = source_node_id("sha256:src33")
    flagged = [{"source_id": sid, "seg_index": 1, "transcriber": "voxtral"},
               {"source_id": sid, "seg_index": 1, "transcriber": "whisper--small"},  # peer row: not a voxtral target
               {"source_id": "S-elsewhere", "seg_index": 0, "transcriber": "voxtral"}]
    assert [(si, seg["index"]) for si, _, seg in chunks_from_census(m, flagged, "voxtral")] == [(0, 1)]
    assert chunks_from_census(m, flagged, "whisper--small") == [(0, m["sources"][0], m["sources"][0]["segments"][1])]
    assert chunks_from_census(m, [], "voxtral") == []


def test_prompt_hash_is_stable_under_trailing_whitespace():
    assert prompt_hash_of("Transcribe.\n") == prompt_hash_of("Transcribe.") != prompt_hash_of("Transcribe!")
    assert prompt_hash_of("x").startswith("sha256:")


def test_rows_from_manifest_and_flagged_chunks_need_no_graph():
    """The qt lane's offline source: census rows straight from the manifest (rendition key =
    (source, segment); degenerate markers + landing provenance ride along), and the jump
    index keyed by (source index, segment index) in manifest order."""
    from cjm_transcription_core.chunk import flagged_chunks, rows_from_manifest
    m = _manifest()
    m["collections"] = [{"title": "GPU MODE"}]
    m["sources"][0]["segments"][0]["transcripts"]["voxtral"]["metadata"] = {"degenerate_tail": {"repeats": 9},
                                                                          "landing": {"producer": "rerun"}}
    rows = rows_from_manifest(m)
    assert len(rows) == 6 and {r["collection"] for r in rows} == {"GPU MODE"}
    r = next(r for r in rows if r["source_index"] == 0 and r["seg_index"] == 1 and r["transcriber"] == "voxtral")
    assert r["words"] == 25000 and r["chars"] == 125000 and r["rendition_id"] == "0:1" and r["config_hash"] == "sha256:cv-old"
    assert r["source_id"] == source_node_id("sha256:src33") and not r["superseded"]
    g = next(r for r in rows if r["source_index"] == 0 and r["seg_index"] == 0 and r["transcriber"] == "voxtral")
    assert g["degenerate"] == 1 and g["producer"] == "rerun"
    flags = flagged_chunks(m)
    assert list(flags) == [(0, 0), (0, 1)]
    assert {f["transcriber"] for f in flags[(0, 0)]} == {"voxtral"} and flags[(0, 0)][0]["reasons"] == ["degenerate"]
    assert {f["transcriber"] for f in flags[(0, 1)]} == {"voxtral", "whisper--small"}
    assert flagged_chunks(m, max_chars=10 ** 9, max_words_per_second=10 ** 9, disagreement_ratio=10 ** 9) == {(0, 0): flags[(0, 0)]}


def test_render_escalation_prompt_fills_context_and_hashes_the_template():
    """Prompt as DATA (f304d31d / 8a9b9639 (3)): the rendered prompt carries the source title,
    collection, chunk range, neighbouring text and the current draft; the hash is the
    TEMPLATE's (one template = one variant identity across chunks), never the rendering's."""
    from cjm_transcription_core.chunk import DEFAULT_ESCALATION_PROMPT, render_escalation_prompt
    m = _manifest()
    m["collections"] = [{"title": "GPU MODE"}]
    m["sources"][0]["segments"][1]["transcripts"]["voxtral"]["text"] = "some " * 400
    out = render_escalation_prompt(m, 0, 1, glossary=["NCCL", "CUTLASS"])
    p = out["prompt"]
    assert "Recording: Lecture 33: Bitblas" in p and "Collection: GPU MODE" in p and "Chunk: 220s-440s" in p
    assert "BEFORE this chunk (another transcriber): fine" in p, "the previous chunk's text (first transcriber with text)"
    assert "AFTER this chunk (another transcriber): (none)" in p
    assert "Current draft of THIS chunk (may be wrong or truncated): ok" in p, "draft = the first transcriber's text"
    assert "Glossary / known terms: NCCL, CUTLASS" in p
    assert out["prompt_hash"] == prompt_hash_of(DEFAULT_ESCALATION_PROMPT) == render_escalation_prompt(m, 1, 0)["prompt_hash"]
    vox = render_escalation_prompt(m, 0, 1, transcriber="voxtral", draft_chars=20)
    assert vox["slots"]["draft_text"].startswith("some some") and len(vox["slots"]["draft_text"]) <= 20
    custom = render_escalation_prompt(m, 0, 1, template="Transcribe {source_title} ({chunk_range}).")
    assert custom["prompt"] == "Transcribe Lecture 33: Bitblas (220s-440s)." and custom["prompt_hash"] != out["prompt_hash"]
    with pytest.raises(ValueError):
        render_escalation_prompt(m, 0, 7)
    with pytest.raises(ValueError):
        render_escalation_prompt(m, 5, 0)
