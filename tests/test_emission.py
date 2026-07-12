"""Tests for cjm_transcription_core.emission — emission payload shape + identity determinism.

Projected from the emission notebook's test cell at the golden-reference flip
(pure; no capabilities involved)."""
import dataclasses

import pytest

from cjm_transcription_core.emission import build_source_emission
from cjm_transcription_core.models import SegmentRecord, SourceResult
from cjm_transcript_graph_schema.schema import (
    audio_rendition_node_id,
    audio_segment_node_id,
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
