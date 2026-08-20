"""Per-segment comparison probe: transcribe ONE VAD-cut segment across every
candidate (capability, MODEL) instance so the operator picks the run's
lightweight/accuracy pair on evidence instead of vibes (work item be4627c7).

Reuses the core pipeline's building blocks verbatim (convert -> VAD ->
boundaries -> cut -> one-segment composition over the candidate instance ids);
capability-side caches make repeat probes — and the eventual full run over the
same source — cheap. Comparison rows carry the transcript text plus the CR-7
empirical profile of each instance (duration mean / GPU + RSS peaks) once the
probe itself has seeded samples, so the "lightweight" judgment gets numbers.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

from cjm_transcription_core.boundaries import compute_segment_boundaries
from cjm_transcription_core.pipeline import (analyze_vad, build_segment_composition,
                                             convert_for_vad, cut_segments, probe_duration,
                                             records_from_composition)


class SegmentProbe:
    """One source's cut segments + cached per-segment comparison results.

    prepare() runs convert -> VAD -> boundaries -> cut ONCE per source (all
    cache-friendly); compare(i) fans the i-th raw segment out to every candidate
    instance as a one-segment composition and caches rows by segment index, so
    walking segments stays interactive and re-visiting one is free. The queue's
    manager reference supplies per-instance CR-7 empirical profiles for the rows.
    """

    def __init__(
        self,
        manager: Any,            # CapabilityManager (loaded instances; empirical profiles)
        queue: Any,              # Started JobQueue
        cfg: Any,                # PipelineConfig (vad/ffmpeg ids, rates, segment cap)
        source_path: str,        # The source to sample
        transcriber_ids: List[str],  # Candidate instance ids to fan out to
        preprocessing_id: Optional[str] = None,  # source_separation instance id (app loads it on first toggle; None = axis unavailable)
    ):
        self.manager = manager
        self.queue = queue
        self.cfg = cfg
        self.source_path = source_path
        self.transcriber_ids = list(transcriber_ids)
        self.preprocessing_id = preprocessing_id
        self.preprocess = False  # The A/B toggle (d gesture): fan out WITH preprocessing?
        self.raw_segments: List[Any] = []
        self.duration = 0.0
        # Caches key on (segment index, preprocess flag): flipping the A/B back
        # is free, and playback follows the toggled rendition automatically.
        self._rows: Dict[Tuple[int, bool], List[Dict[str, Any]]] = {}
        self._wavs: Dict[Tuple[int, bool], str] = {}  # -> model-input WAV (playback)
        self.active_comp_id = None  # In-flight probe composition (escape-cancel target)

    async def prepare(self) -> int:  # Segment count after cutting
        """Convert -> VAD -> boundaries -> cut the source once; return segment count."""
        vad_audio = await convert_for_vad(
            self.queue, self.cfg.ffmpeg_capability, self.source_path,
            sample_rate=self.cfg.sample_rate, channels=self.cfg.channels)
        chunks, duration = await analyze_vad(self.queue, self.cfg.vad_capability, vad_audio)
        if duration <= 0:
            duration = await probe_duration(self.queue, self.cfg.ffmpeg_capability,
                                            self.source_path)
        self.duration = duration
        boundaries = compute_segment_boundaries(chunks, self.cfg.max_segment_duration,
                                                duration)
        self.raw_segments, _ = await cut_segments(self.queue, self.cfg.ffmpeg_capability,
                                                  self.source_path, boundaries)
        return len(self.raw_segments)

    def wav_path(self, index: int) -> Optional[str]:
        """The i-th segment's model-input WAV under the CURRENT A/B toggle, once
        compare(i) has folded records for that state (vocals-isolated when ON)."""
        return self._wavs.get((index, self.preprocess))

    def profile(self, instance_id: str) -> Dict[str, Any]:
        """The instance's CR-7 empirical record fields worth showing (may be {})."""
        inst = self.manager.get_instance(instance_id)
        store = getattr(self.manager, "empirical_store", None)
        if inst is None or store is None or not getattr(inst, "config_hash", None):
            return {}
        try:
            rec = store.get_record(inst.instance_id, inst.config_hash)
        except Exception:
            return {}
        if rec is None:
            return {}
        return {"duration_s_mean": rec.duration_seconds_mean,
                "gpu_mb_peak": rec.gpu_memory_mb_peak_max,
                "rss_mb_peak": rec.memory_mb_peak_max,
                "samples": rec.sample_count}

    async def cancel_active(self) -> bool:  # True when a cancel was dispatched
        """Cancel the in-flight probe composition, if any (the escape gesture).

        Cooperative substrate cancellation with force fallback; the waiting
        compare() then surfaces the cancelled composition as a status error
        instead of leaving the operator staring at a spinner (drive-1 finding
        30057f10 — pairs with the queue's never-fits fail-fast 532ea1da).
        """
        comp_id = self.active_comp_id
        if comp_id is None:
            return False
        try:
            await self.queue.cancel_composition(comp_id)
            return True
        except Exception:
            return False

    async def compare(self, index: int) -> List[Dict[str, Any]]:
        """Transcribe segment `index` across every candidate; one row per candidate.

        Fans out under the CURRENT preprocessing toggle — the same composition
        shape the headless run uses (stage 8: preprocess -> convert -> T×
        transcribe), so the A/B rows predict exactly what --preprocessing-capability
        would do to the full run."""
        key = (index, self.preprocess)
        if key in self._rows:
            return self._rows[key]
        seg = self.raw_segments[index]
        comp, metas = build_segment_composition(
            [seg], f"probe-{int(time.time())}", 0,
            self.cfg.ffmpeg_capability, self.transcriber_ids,
            sample_rate=self.cfg.sample_rate, channels=self.cfg.channels,
            preprocessing_capability=(self.preprocessing_id if self.preprocess
                                      else None))
        comp_id = await self.queue.submit_composition(comp)
        self.active_comp_id = comp_id
        try:
            crun = await self.queue.wait_for_composition(comp_id)
        finally:
            self.active_comp_id = None
        records = records_from_composition(crun, metas)
        rec = records[0] if records else None
        if rec is not None and rec.model_input_path:
            # What the candidates heard — the playback source (kit ChunkPlayer
            # plays this, not the source media; correction-TUI principle).
            self._wavs[key] = rec.model_input_path
        rows: List[Dict[str, Any]] = []
        for tid in self.transcriber_ids:
            tr = (rec.transcripts.get(tid) if rec is not None else None) or {}
            text = tr.get("text") or ""
            rows.append({"instance_id": tid, "text": text, "chars": len(text),
                         "profile": self.profile(tid)})
        self._rows[key] = rows
        return rows
