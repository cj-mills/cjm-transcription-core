"""Tests for cjm_transcription_core.boundaries — wall-clock-aware segment boundary computation.

Projected from the boundaries notebook's behavioral-check cell at the golden-reference
flip (mirrors the validated audio-segment semantics)."""
from cjm_transcription_core.boundaries import compute_segment_boundaries


def test_short_audio_single_covering_segment():
    assert compute_segment_boundaries([{"start": 1.0, "end": 5.0}], 300.0, 28.0) == [
        {"start": 0.0, "end": 28.0}
    ]


def test_no_chunks_single_covering_segment():
    assert compute_segment_boundaries([], 300.0, 28.0) == [{"start": 0.0, "end": 28.0}]


def test_zero_duration_no_segments():
    assert compute_segment_boundaries([], 300.0, 0.0) == []


def test_cut_at_silence_gap_midpoint():
    # Two chunks straddling the cap -> cut at the silence-gap midpoint
    chunks = [{"start": 0.0, "end": 100.0}, {"start": 120.0, "end": 200.0}]
    b = compute_segment_boundaries(chunks, 150.0, 200.0)
    assert b == [{"start": 0.0, "end": 110.0}, {"start": 110.0, "end": 200.0}], b


def test_non_final_wall_clock_invariant():
    # Non-final wall-clock invariant holds on a longer synthetic chunk train
    chunks = [{"start": float(i * 10), "end": float(i * 10 + 8)} for i in range(60)]
    b = compute_segment_boundaries(chunks, 60.0, 600.0)
    assert all((s["end"] - s["start"]) <= 60.0 for s in b[:-1]), b
    assert b[0]["start"] == 0.0
    assert b[-1]["end"] == 600.0
