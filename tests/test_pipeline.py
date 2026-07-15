"""Tests for cjm_transcription_core.pipeline — pure-logic checks (no capabilities involved).

Projected from the pipeline notebook's three check cells at the golden-reference flip:
pure-logic smoke checks, composition builder/folder checks, preprocessing composition."""
import pytest

from cjm_capability_primitives.media_processing import MediaArtifactResult, MediaSegment
from cjm_capability_primitives.transcription import TranscriptionResult
from cjm_capability_primitives.vad import TimeRange, VADResult
from cjm_substrate.core.ports import NodeState, OutputRef, new_composition_run

from cjm_transcription_core.models import SegmentRecord
from cjm_transcription_core.pipeline import (
    build_segment_composition,
    collect_capability_info,
    confirm_seam,
    normalize_vad_result,
    records_from_composition,
    tier1_segment_checks,
    tier1_transcript_checks,
)

RAW_SEGMENTS = [
    MediaSegment(index=0, output_path="/seg0.flac", start=0.0, end=280.0, duration=280.0),
    MediaSegment(index=1, output_path="/seg1.flac", start=280.0, end=560.0, duration=280.0),
]


def test_normalize_vad_result_typed():
    # normalize_vad_result consumes the TYPED result (stage-8 VADResult);
    # ordering still normalizes, duration still reads from metadata.
    chunks, dur = normalize_vad_result(VADResult(
        ranges=[TimeRange(start=5.0, end=9.0), TimeRange(start=0.5, end=2.0)],
        metadata={"duration": 28.0},
    ))
    assert chunks == [{"start": 0.5, "end": 2.0}, {"start": 5.0, "end": 9.0}], chunks
    assert dur == 28.0


def test_tier1_checks():
    # tier-1 checks fire on the right shapes (0.2.0 per-transcriber transcripts)
    assert tier1_segment_checks([], 300.0, 0) != []
    assert tier1_segment_checks([{"start": 0.0, "end": 28.0}], 300.0, 3) == []
    empty = SegmentRecord(0, 0.0, 40.0, 40.0, "a", "b",
                          transcripts={"whisper": {"job_id": "j", "text": "", "metadata": {}}})
    ok = SegmentRecord(0, 0.0, 28.0, 28.0, "a", "b",
                       transcripts={"whisper": {"job_id": "j", "text": "plenty of text here", "metadata": {}}})
    assert tier1_transcript_checks([empty]) != []
    assert tier1_transcript_checks([ok]) == []
    # per-transcriber warnings name the transcriber
    dual = SegmentRecord(0, 0.0, 40.0, 40.0, "a", "b", transcripts={
        "whisper": {"job_id": "j1", "text": "plenty of text here too", "metadata": {}},
        "voxtral": {"job_id": "j2", "text": "", "metadata": {}},
    })
    ws = tier1_transcript_checks([dual])
    assert len(ws) == 1 and "[voxtral]" in ws[0]


def test_confirm_seam_headless():
    assert confirm_seam("boundary-review", ["x"], [], assume_yes=True) is True


def test_composition_builder_and_folder():
    comp, metas = build_segment_composition(RAW_SEGMENTS, "runX", 0, "ffmpeg", ["whisper"])
    assert len(comp.nodes) == 4 and len(metas) == 2
    # Each pipe: transcribe bound to its OWN convert's output_path.
    assert comp.nodes[1].kwargs["audio"] == OutputRef("convert_0000", "output_path")
    assert comp.nodes[3].kwargs["audio"] == OutputRef("convert_0001", "output_path")
    # Per-item provenance kwargs are STATIC (host-computed at construction).
    assert comp.nodes[1].kwargs["job_id"] == "runX_src0_seg0000_t0"
    assert comp.nodes[3].kwargs["source_start_time"] == 280.0
    # The two pipes are independent: converts have no deps; fan-out parallelizes.
    run = new_composition_run(comp, "r")
    assert run.ready_nodes() == ["convert_0000", "convert_0001"]

    # Fold a completed run back into records (per-transcriber transcripts).
    run.record_result("convert_0000", NodeState.completed, result=MediaArtifactResult(output_path="/c0.wav"))
    run.record_result("convert_0001", NodeState.completed, result=MediaArtifactResult(output_path="/c1.wav"))
    run.record_result("transcribe_t0_0000", NodeState.completed,
                      result=TranscriptionResult(text="hello", metadata={"m": 1}))
    run.record_result("transcribe_t0_0001", NodeState.completed,
                      result=TranscriptionResult(text="world", metadata={}))
    run.status = NodeState.completed
    recs = records_from_composition(run, metas)
    assert [r.transcripts["whisper"]["text"] for r in recs] == ["hello", "world"]
    assert recs[0].model_input_path == "/c0.wav"
    assert recs[1].transcripts["whisper"]["job_id"] == "runX_src0_seg0001_t0"
    assert recs[0].duration == 280.0


def test_dual_transcriber_fan_out():
    # DUAL-transcriber fan-out (stage 5: the named parallel-port adopter): one
    # convert per segment, one transcribe node per transcriber off the same output.
    comp2, metas2 = build_segment_composition(RAW_SEGMENTS, "runX", 0, "ffmpeg", ["whisper", "voxtral"])
    assert len(comp2.nodes) == 6
    assert comp2.nodes[1].kwargs["audio"] == comp2.nodes[2].kwargs["audio"] == OutputRef("convert_0000", "output_path")
    assert comp2.nodes[1].kwargs["job_id"] == "runX_src0_seg0000_t0"
    assert comp2.nodes[2].kwargs["job_id"] == "runX_src0_seg0000_t1"
    assert metas2[0]["transcribe_nodes"] == {"whisper": "transcribe_t0_0000", "voxtral": "transcribe_t1_0000"}
    run2 = new_composition_run(comp2, "r2")
    assert run2.ready_nodes() == ["convert_0000", "convert_0001"]
    run2.record_result("convert_0000", NodeState.completed, result=MediaArtifactResult(output_path="/c0.wav"))
    run2.record_result("convert_0001", NodeState.completed, result=MediaArtifactResult(output_path="/c1.wav"))
    for nid, txt in [("transcribe_t0_0000", "w0"), ("transcribe_t1_0000", "v0"),
                     ("transcribe_t0_0001", "w1"), ("transcribe_t1_0001", "v1")]:
        run2.record_result(nid, NodeState.completed, result=TranscriptionResult(text=txt, metadata={}))
    run2.status = NodeState.completed
    recs2 = records_from_composition(run2, metas2)
    assert recs2[0].transcripts["whisper"]["text"] == "w0" and recs2[0].transcripts["voxtral"]["text"] == "v0"
    assert recs2[1].transcripts["voxtral"]["text"] == "v1"


def test_failed_run_surfaces_failed_nodes():
    comp, metas = build_segment_composition(RAW_SEGMENTS, "runX", 0, "ffmpeg", ["whisper"])
    bad = new_composition_run(comp, "r3")
    bad.record_result("convert_0000", NodeState.failed)
    bad.status = NodeState.failed
    with pytest.raises(RuntimeError) as ei:
        records_from_composition(bad, metas)
    assert "convert_0000" in str(ei.value)


def test_preprocessing_composition():
    # Stage 8: opt-in preprocessing inserts a source_separation node FIRST per segment,
    # and the model-input convert REBINDS to the preprocessing output (full-band raw
    # -> separate_vocals -> convert -> transcribe). No-preprocessing path unchanged.
    compp, metasp = build_segment_composition(
        RAW_SEGMENTS, "runP", 0, "ffmpeg", ["whisper"],
        preprocessing_capability="cjm-capability-demucs")
    # per segment now: separate -> convert -> transcribe == 3 nodes x 2 segs = 6
    assert len(compp.nodes) == 6
    sep0, conv0, tr0 = compp.nodes[0], compp.nodes[1], compp.nodes[2]
    # the separate node routes through the source_separation task channel on raw audio
    assert sep0.task_name == "source_separation" and sep0.method == "separate_vocals"
    assert sep0.kwargs["audio"] == "/seg0.flac"
    assert sep0.control == {"force": False}
    # the convert input REBOUND to the separation output (NOT the raw seg path)
    assert conv0.kwargs["input_path"] == OutputRef("separate_0000", "output_path")
    # transcribe still binds to the convert output (vocals -> convert -> transcribe)
    assert tr0.kwargs["audio"] == OutputRef("convert_0000", "output_path")
    # the separate nodes have no deps -> they run first
    runp = new_composition_run(compp, "rp")
    assert runp.ready_nodes() == ["separate_0000", "separate_0001"]
    assert metasp[0]["separate_node"] == "separate_0000"
    # default (no preprocessing) leaves the convert reading the raw segment + records None
    compd, metasd = build_segment_composition(RAW_SEGMENTS, "runD", 0, "ffmpeg", ["whisper"])
    assert compd.nodes[0].kwargs["input_path"] == "/seg0.flac"  # convert reads raw seg directly
    assert metasd[0]["separate_node"] is None


def test_collect_capability_info_multi_instance():
    # CR-10 named (capability, MODEL) instances live in manager.instances only —
    # provenance must resolve them through instance.capability_name instead of
    # silently dropping the transcriber from the run manifest (db200725).
    class _Meta:
        def __init__(self, name):
            self.name = name
            self.version = "1.2.3"
            self.manifest = {"db_path": None}

    class _Proxy:
        def __init__(self, config):
            self._config = config

        def get_current_config(self):
            return self._config

    class _Inst:
        def __init__(self, capability_name, proxy):
            self.capability_name = capability_name
            self.proxy = proxy

    class _Manager:
        def __init__(self):
            tiny = _Proxy({"model": "tiny"})
            large = _Proxy({"model": "large-v3"})
            self.capabilities = {"cjm-capability-whisper": _Meta("cjm-capability-whisper")}
            self.instances = {
                "cjm-capability-whisper": _Inst("cjm-capability-whisper", tiny),
                "whisper-large": _Inst("cjm-capability-whisper", large),
            }

        def get_capability(self, name_or_id):
            return self.instances[name_or_id].proxy

    info = collect_capability_info(_Manager(), ["cjm-capability-whisper", "whisper-large", "ghost"])
    # both instances recorded under their INSTANCE ids, named one resolves the meta
    assert set(info) == {"cjm-capability-whisper", "whisper-large"}
    assert info["whisper-large"]["name"] == "cjm-capability-whisper"
    assert info["whisper-large"]["config"] == {"model": "large-v3"}
    assert info["cjm-capability-whisper"]["config"] == {"model": "tiny"}
    # distinct effective configs hash distinctly (Transcript identity input)
    assert info["whisper-large"]["config_hash"] != info["cjm-capability-whisper"]["config_hash"]
