"""Tests for cjm_transcription_core.emission — emission payload shape + identity determinism.

Projected from the emission notebook's test cell at the golden-reference flip
(pure; no capabilities involved)."""
import dataclasses

import pytest

from cjm_transcription_core.emission import build_collection_emission, build_source_emission
from cjm_transcription_core.models import CollectionDecl, SegmentRecord, SourceResult
from cjm_transcript_graph_schema.schema import (
    audio_rendition_node_id,
    audio_segment_node_id,
    collection_node_id,
    source_node_id,
    transcript_node_id,
)


def _fixture_source():
    recs = [
        SegmentRecord(index=0, start=0.0, end=280.0, duration=280.0,
                      segment_path="/cuts/s0.mp3", model_input_path="/cache/s0.wav",
                      model_input_hash="sha256:wav0",
                      transcripts={"whisper": {"job_id": "j0w", "text": "hello", "metadata": {}},
                                   "voxtral": {"job_id": "j0v", "text": "hullo", "metadata": {}}}),
        SegmentRecord(index=1, start=280.0, end=560.0, duration=280.0,
                      segment_path="/cuts/s1.mp3", model_input_path="/cache/s1.wav",
                      model_input_hash="sha256:wav1",
                      transcripts={"whisper": {"job_id": "j1w", "text": "world", "metadata": {}},
                                   "voxtral": {"job_id": "j1v", "text": "wurld", "metadata": {}}}),
    ]
    src = SourceResult(source_path="/media/ep1.mp3", duration=560.0, vad_chunk_count=99,
                       batch_key="bk", content_hash="sha256:src", segments=recs)
    hashes = {"whisper": "sha256:cfgw", "voxtral": "sha256:cfgv"}
    return src, hashes


def test_payload_shape():
    src, hashes = _fixture_source()
    nodes, edges, ids = build_source_emission(src, hashes)
    # 1 Source + 2 AudioSegment + 2 AudioRendition + 2x2 Transcript
    assert len(nodes) == 9
    labels = [n["label"] for n in nodes]
    assert labels.count("Source") == 1 and labels.count("AudioSegment") == 2
    assert labels.count("AudioRendition") == 2 and labels.count("Transcript") == 4
    # spine: 1 STARTS_WITH + 2 PART_OF + 1 NEXT; plus 2 rendition + 4 transcript DERIVED_FROM = 6
    rels = [e["relation_type"] for e in edges]
    assert rels.count("STARTS_WITH") == 1 and rels.count("PART_OF") == 2
    assert rels.count("NEXT") == 1 and rels.count("DERIVED_FROM") == 6
    # AudioSegment is a hashless boundary (model-input moved to the rendition); raw chain -> is_raw rendition
    assert all(n["sources"] == [] and "model_input_path" not in n["properties"]
               for n in nodes if n["label"] == "AudioSegment")
    assert all(n["properties"]["is_raw"] is True and n["sources"][0]["content_hash"].startswith("sha256:")
               for n in nodes if n["label"] == "AudioRendition")


def test_deterministic_ids_recomputable_from_manifest():
    src, hashes = _fixture_source()
    nodes, edges, ids = build_source_emission(src, hashes)
    assert ids["source"] == source_node_id("sha256:src")
    a0 = audio_segment_node_id(ids["source"], 0.0, 280.0)
    assert ids["audio_segments"][0] == a0
    r0 = audio_rendition_node_id(a0, [])  # raw rendition
    assert ids["renditions"][0] == r0
    assert ids["transcripts"]["whisper"][0] == transcript_node_id(r0, "whisper", "sha256:cfgw")
    # re-build -> byte-identical id sets (emission idempotency precondition)
    nodes2, edges2, ids2 = build_source_emission(src, hashes)
    assert [n["id"] for n in nodes2] == [n["id"] for n in nodes]
    assert [e["id"] for e in edges2] == [e["id"] for e in edges]


def test_preprocessing_chain_distinct_renditions_coexist():
    src, hashes = _fixture_source()
    nodes, edges, ids = build_source_emission(src, hashes)
    a0 = audio_segment_node_id(ids["source"], 0.0, 280.0)
    chain = ["source_separation:cjm-capability-demucs@cfg123"]
    nodes_p, edges_p, ids_p = build_source_emission(src, hashes, chain=chain)
    assert ids_p["audio_segments"] == ids["audio_segments"]  # boundary shared across renditions
    assert ids_p["renditions"] != ids["renditions"]          # vocals renditions are distinct nodes
    assert ids_p["transcripts"]["whisper"] != ids["transcripts"]["whisper"]
    assert ids_p["renditions"][0] == audio_rendition_node_id(a0, chain)
    assert all(n["properties"]["is_raw"] is False and n["properties"]["preprocessing"] == chain[0]
               for n in nodes_p if n["label"] == "AudioRendition")
    # raw + vocals payloads share zero rendition ids -> they can land in ONE graph without collision
    assert not (set(ids["renditions"]) & set(ids_p["renditions"]))


def test_identity_guards_fire_loudly():
    src, hashes = _fixture_source()
    with pytest.raises(ValueError):
        build_source_emission(dataclasses.replace(src, content_hash=""), hashes)
    no_hash = dataclasses.replace(
        src, segments=[dataclasses.replace(src.segments[0], model_input_hash="")])
    with pytest.raises(ValueError):
        build_source_emission(no_hash, hashes)


def test_collection_emission_payload():
    cid = collection_node_id("Hardcore History")
    decl = CollectionDecl(title="Hardcore History", status="proposed", actor="cli:transcribe",
                          member_paths=["/media/ep1.mp3", "/media/ep2.mp3", "/media/gone.mp3"],
                          ordered=True)
    s1, s2 = source_node_id("sha256:ep1"), source_node_id("sha256:ep2")
    path_map = {"/media/ep1.mp3": s1, "/media/ep2.mp3": s2}

    # only COMPLETED sources file in; the unresolved member is skipped
    nodes, edges, ids = build_collection_emission(decl, path_map)
    assert ids == {"collection": cid, "members": [s1, s2]}
    assert nodes[0]["label"] == "Collection"
    assert nodes[0]["properties"]["status"] == "proposed"
    assert nodes[0]["properties"]["root_kind"] == "asserted"
    rels = sorted(e["relation_type"] for e in edges)
    assert rels == ["NEXT", "PART_OF", "PART_OF", "STARTS_WITH"], "ordered decl = full spine"

    # unordered declaration files membership without fabricating sequence
    _, loose_edges, _ = build_collection_emission(
        CollectionDecl(title="Hardcore History", member_paths=["/media/ep2.mp3"]), path_map)
    assert [e["relation_type"] for e in loose_edges] == ["PART_OF"]
    assert loose_edges[0]["target_id"] == cid, "late member attaches to the SAME node"

    # no resolvable members = build nothing (capture never invents an empty collection)
    empty = build_collection_emission(
        CollectionDecl(title="Empty", member_paths=["/media/gone.mp3"]), path_map)
    assert empty == ([], [], {"collection": None, "members": []})


def test_structure_entries_from_map_folds_cells_and_refuses_empty_rows():
    """The document shape (cells flat per row) folds into op entries: `source_id`
    and `evidence` lift out, `_`-prefixed commentary never lands, every other
    key is a cell; a row without cells or without a source refuses."""
    from cjm_transcription_core.curation import structure_entries_from_map

    doc = {"_": "commentary", "collection_id": "coll-1",
           "entries": [{"file": 4, "source_id": "src-4", "kind": "chapter", "part": 1,
                        "part_title": "School", "chapter": 1, "title": "Seven Lessons",
                        "evidence": ["strata:'Part 1. School. Chapter 1.'", "toc"], "_note": "x"},
                       {"file": 2, "source_id": "src-2", "kind": "front-matter", "unit": "foreword"}]}
    entries = structure_entries_from_map(doc)
    assert [e["source_id"] for e in entries] == ["src-4", "src-2"]
    assert entries[0]["structure"] == {"file": 4, "kind": "chapter", "part": 1, "part_title": "School",
                                       "chapter": 1, "title": "Seven Lessons"}
    assert entries[0]["evidence"] == ["strata:'Part 1. School. Chapter 1.'", "toc"]
    assert entries[1]["structure"] == {"file": 2, "kind": "front-matter", "unit": "foreword"}
    assert entries[1]["evidence"] == []
    with pytest.raises(ValueError):
        structure_entries_from_map({"entries": [{"kind": "chapter"}]})
    with pytest.raises(ValueError):
        structure_entries_from_map({"entries": [{"source_id": "src-9", "evidence": ["toc"]}]})


def test_declare_structure_journals_property_merges(tmp_path):
    """`declare_structure` lands one `work_structure` property merge per entry
    (cells + evidence together on the Source) through the collection-curation
    op shape — act `declare-structure`, no deletes, no wires — and appends the
    op verbatim to the sidecar journal so a rebuild replays the same merges."""
    import asyncio
    import json
    from types import SimpleNamespace

    from cjm_substrate.core.queue import JobStatus
    from cjm_transcription_core.curation import declare_structure

    class FakeQueue:
        def __init__(self):
            self.submitted = []

        async def submit(self, graph_id, **kw):
            self.submitted.append((graph_id, kw))
            return "j1"

        async def wait_for_job(self, jid):
            return SimpleNamespace(status=JobStatus.completed, result=True, error=None)

    q = FakeQueue()
    journal = tmp_path / "context_graph.writes.jsonl"
    entries = [{"source_id": "src-4", "structure": {"kind": "chapter", "part": 1, "chapter": 1},
                "evidence": ["strata:readout", "toc"]},
               {"source_id": "src-5", "structure": {"kind": "chapter", "part": 1, "chapter": 2},
                "evidence": ["toc"]}]
    op = asyncio.run(declare_structure(q, "g", entries, "human:tester",
                                       journal_path=str(journal), collection_id="coll-1"))
    assert op["verb"] == "collection-curation"
    assert op["args"] == {"act": "declare-structure", "collection_id": "coll-1", "sources": 2}
    assert op["deletes"] == {"edge_ids": [], "node_ids": []}
    assert op["wires"] == {"nodes": [], "edges": []}
    assert op["updates"] == [
        {"id": "src-4", "properties": {"work_structure": {"kind": "chapter", "part": 1, "chapter": 1,
                                                          "evidence": ["strata:readout", "toc"]}}},
        {"id": "src-5", "properties": {"work_structure": {"kind": "chapter", "part": 1, "chapter": 2,
                                                          "evidence": ["toc"]}}}]
    # Applied as update_node merges, one per entry, on the graph id given.
    assert [(g, kw["method"], kw["node_id"]) for g, kw in q.submitted] == [
        ("g", "update_node", "src-4"), ("g", "update_node", "src-5")]
    assert q.submitted[0][1]["properties"] == op["updates"][0]["properties"]
    # Journaled verbatim (one line, the whole op).
    lines = [json.loads(l) for l in journal.read_text().splitlines() if l.strip()]
    assert len(lines) == 1
    assert lines[0]["verb"] == "collection-curation"
    assert lines[0]["updates"] == op["updates"]
    # An entry without cells refuses before anything is applied.
    q2 = FakeQueue()
    with pytest.raises(ValueError):
        asyncio.run(declare_structure(q2, "g", [{"source_id": "src-9", "structure": {}}], "human:tester"))
    assert q2.submitted == []


def test_add_and_retract_reference_journal_wires_and_a_cascade_delete(tmp_path, monkeypatch):
    """`add_reference` lands ONE Reference node + ONE HAS_REFERENCE edge through the
    collection-curation op shape (act `add-reference`; no deletes, no updates), with a
    (source, target)-derived node id so the same link re-added is the same node;
    `retract_reference` is the compensating act (a cascade node delete). Both append
    verbatim to the sidecar journal. Refusals: no target, no label. The apply leg is
    recorded, not run (extend-verify needs a real graph; the op SHAPE is the contract)."""
    import asyncio
    import json

    from cjm_transcription_core import curation
    from cjm_transcription_core.curation import add_reference, retract_reference
    from cjm_transcript_graph_schema.schema import TranscriptGraphLabels

    applied = []

    async def record(queue, graph_id, op):
        applied.append((graph_id, op))
    monkeypatch.setattr(curation, "apply_curation", record)

    journal = tmp_path / "context_graph.writes.jsonl"
    op = asyncio.run(add_reference(None, "g", "src-4", label="Dumbing Us Down (publisher page)",
                                   url="https://newsociety.com/book/dumbing-us-down-25th-anniversary-edition/",
                                   role="cited-work", actor="human:tester", journal_path=str(journal)))
    assert op["verb"] == "collection-curation" and op["args"]["act"] == "add-reference"
    assert op["args"]["source_id"] == "src-4" and op["args"]["role"] == "cited-work"
    assert op["deletes"] == {"edge_ids": [], "node_ids": []} and op["updates"] == []
    ref_id = op["reference_id"]
    assert op["args"]["reference_id"] == ref_id
    [node] = op["wires"]["nodes"]
    [edge] = op["wires"]["edges"]
    assert node["id"] == ref_id and node["label"] == TranscriptGraphLabels.REFERENCE == "Reference"
    assert node["properties"] == {"source_id": "src-4", "label": "Dumbing Us Down (publisher page)",
                                  "url": "https://newsociety.com/book/dumbing-us-down-25th-anniversary-edition/",
                                  "notes_slug": "", "role": "cited-work", "added_by": "human:tester"}
    assert (edge["source_id"], edge["target_id"], edge["relation_type"]) == ("src-4", ref_id, "HAS_REFERENCE")
    assert applied and applied[0][0] == "g" and applied[0][1]["args"]["act"] == "add-reference"
    # Same (source, target) -> same id; a different target -> a different node.
    op2 = asyncio.run(add_reference(None, "g", "src-4", label="renamed", url=node["properties"]["url"], actor="human:tester"))
    assert op2["reference_id"] == ref_id
    op3 = asyncio.run(add_reference(None, "g", "src-4", label="Notes on Dumbing Us Down",
                                    notes_slug="dumbing-us-down/ch01-notes", url="https://example.org/fallback",
                                    role="related-notes", actor="human:tester"))
    assert op3["reference_id"] != ref_id
    assert op3["wires"]["nodes"][0]["properties"]["notes_slug"] == "dumbing-us-down/ch01-notes"
    # Retract: a cascade node delete, journaled.
    rop = asyncio.run(retract_reference(None, "g", ref_id, actor="human:tester", journal_path=str(journal)))
    assert rop["args"] == {"act": "retract-reference", "reference_id": ref_id}
    assert rop["deletes"] == {"edge_ids": [], "node_ids": [ref_id]} and rop["wires"] == {"nodes": [], "edges": []}
    lines = [json.loads(l) for l in journal.read_text().splitlines() if l.strip()]
    assert [l["args"]["act"] for l in lines] == ["add-reference", "retract-reference"]
    assert lines[0]["wires"]["nodes"][0]["id"] == ref_id
    # Refusals before anything is applied.
    n_applied = len(applied)
    with pytest.raises(ValueError):
        asyncio.run(add_reference(None, "g", "src-4", label="x", actor="human:tester"))
    with pytest.raises(ValueError):
        asyncio.run(add_reference(None, "g", "src-4", label="  ", url="https://x", actor="human:tester"))
    assert len(applied) == n_applied
