"""Wall-clock-aware segment boundary computation: group VAD speech chunks into segments cut at silence-gap midpoints. Pure logic — no capability calls. Final home of the algorithm originally validated in cjm-transcription-audio-segment's AudioSegmentService.compute_segment_boundaries (that library is retired to cj-mills_deferred/)."""

from typing import Dict, List, Optional


def compute_segment_boundaries(
    vad_chunks: List[Dict[str, float]],  # [{start, end, ...}, ...] sorted by start
    max_segment_duration: float,         # Target max wall-clock segment length in seconds
    audio_duration: float,               # Full audio duration in seconds
) -> List[Dict[str, float]]:  # [{start, end}, ...] covering [0, audio_duration]
    """Group VAD chunks into segments cut at silence-gap midpoints.

    **Wall-clock-aware, pre-emptive cuts.** `max_segment_duration` caps the
    wall-clock duration of each output segment (not the speech-only duration
    within it) — matching the downstream forced-alignment constraint, which
    operates on the resulting audio file's length.

    Algorithm:
      1. If audio_duration <= max_segment_duration or no chunks: single segment
         covering [0, audio_duration].
      2. Walk chunks sequentially. For each chunk, check whether accepting it
         would push the in-progress segment's wall-clock duration over max. If
         so AND we already have content: cut **before** this chunk at the
         silence-gap midpoint between the previous chunk's end and this chunk's
         start (chunks that abut with no gap cut exactly at the previous
         chunk's end).
      3. The final segment extends to audio_duration.

    **Wall-clock invariant.** Every NON-FINAL segment's wall-clock duration is
    <= max_segment_duration. The final segment may exceed max only because it
    extends to audio_duration to cover trailing silence. A single VAD chunk
    whose own duration exceeds max forms a segment of its native length —
    speech is never split mid-chunk.
    """
    if audio_duration <= 0:
        return []
    if audio_duration <= max_segment_duration or not vad_chunks:
        return [{"start": 0.0, "end": float(audio_duration)}]

    boundaries: List[Dict[str, float]] = []
    segment_start = 0.0
    prev_chunk_end: Optional[float] = None

    for chunk in vad_chunks:
        chunk_start = float(chunk["start"])
        chunk_end = float(chunk["end"])

        # Pre-emptive cut: if accepting this chunk would push the segment's
        # wall-clock duration over max AND we have prior content, cut now.
        if prev_chunk_end is not None and (chunk_end - segment_start) > max_segment_duration:
            cut_point = (
                (prev_chunk_end + chunk_start) / 2.0
                if chunk_start > prev_chunk_end
                else prev_chunk_end
            )
            boundaries.append({"start": segment_start, "end": cut_point})
            segment_start = cut_point

        prev_chunk_end = chunk_end

    boundaries.append({"start": segment_start, "end": float(audio_duration)})
    return boundaries
