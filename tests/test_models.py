"""Tests for cjm_transcription_core.models — run configuration + run-manifest containers.

Projected from the models notebook's shape-check demo cell at the golden-reference
flip (no capabilities involved)."""
import time

from cjm_transcription_core.models import (
    PipelineConfig,
    RunManifest,
    SegmentRecord,
    SourceResult,
    new_run_id,
)


def test_manifest_shape_and_config_defaults():
    cfg = PipelineConfig()
    m = RunManifest(run_id=new_run_id(), created_at=time.time(), config=cfg.to_dict())
    d = m.to_dict()
    assert d["format"] == "cjm-transcription-core/run-manifest"
    assert d["version"] == "0.3.0"
    assert d["config"]["max_segment_duration"] == 220.0
    assert d["config"]["transcriber_capabilities"] == ["cjm-capability-whisper"]
    assert d["graph"] is None
    # preprocessing is OFF by default; the family-agnostic slot defaults to source_separation
    assert d["config"]["preprocessing_capability"] is None
    assert d["config"]["preprocessing_task"] == "source_separation"
    assert d["config"]["preprocessing_method"] == "separate_vocals"


def test_segment_record_per_transcriber_shape():
    # 0.2.0 per-transcriber segment shape
    rec = SegmentRecord(index=0, start=0.0, end=10.0, duration=10.0,
                        segment_path="/cuts/s0.mp3", model_input_path="/cache/s0.wav",
                        model_input_hash="sha256:wav",
                        transcripts={"whisper": {"job_id": "j1", "text": "hi", "metadata": {}}})
    d = rec.to_dict()
    assert d["transcripts"]["whisper"]["text"] == "hi"
    assert d["model_input_hash"] == "sha256:wav"


def test_source_result_chain_and_hash():
    rec = SegmentRecord(index=0, start=0.0, end=10.0, duration=10.0,
                        segment_path="/cuts/s0.mp3", model_input_path="/cache/s0.wav")
    src = SourceResult(source_path="/a.mp3", duration=10.0, vad_chunk_count=3,
                       batch_key="bk", content_hash="sha256:src", segments=[rec])
    assert src.to_dict()["content_hash"] == "sha256:src"
    assert src.to_dict()["graph"] is None
    # chain defaults to [] (raw convert-only); a preprocessed run records its chain
    assert src.to_dict()["chain"] == []
    src_vox = SourceResult(source_path="/a.mp3", duration=10.0, vad_chunk_count=3, batch_key="bk",
                           content_hash="sha256:src", segments=[rec],
                           chain=["source_separation:cjm-capability-demucs@cfg"])
    assert src_vox.to_dict()["chain"] == ["source_separation:cjm-capability-demucs@cfg"]


def test_manifest_save_round_trip(tmp_path):
    m = RunManifest(run_id=new_run_id(), created_at=time.time(),
                    config=PipelineConfig().to_dict())
    out = m.save(tmp_path / "runs" / "manifest.json")
    assert out.exists()
    import json
    loaded = json.loads(out.read_text())
    assert loaded["run_id"] == m.run_id
