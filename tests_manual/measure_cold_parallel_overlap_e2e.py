#!/usr/bin/env python
"""Cold-run parallel-overlap measurement (stage-5 closeout; the carried G11
discipline: parallelism claims need WALL-CLOCK assertions, not green runs).

Runs the full pipeline over the given FRESH audio sources with the
dual-transcriber fan-out and a raised ffmpeg SG-33 cap, then measures
per-lane overlap from in-process Job records (started_at/completed_at):

  - transcriber-A ∥ transcriber-B co-run time (different instances;
    empirical-admission governed)
  - ffmpeg same-instance overlap under the raised cap (subprocess-backed
    workers parallelize; model workers stay serial-per-instance)
  - per-lane summed duration vs composition wall time

Sources must be COLD (content not in the capability caches) for the
measurement to be honest — cache hits hide the wall-clock signal (G11).
Inspect `.cjm/capability_configs.db` for leftover persisted configs BEFORE any
cold run (the I8 lesson).

As-measured baseline (2026-06-11, RTX 4090, HH1 16.8min + HH2 35min):
  wall 206.9s vs summed 367.4s (1.78x); whisper∥voxtral co-run 157.2s =
  100% of the shorter lane; ffmpeg cap=4 max in-flight 4.

Run from the repo root in the cjm-transcription-core env:
    python tests_manual/measure_cold_parallel_overlap_e2e.py audio1.mp3 [audio2.mp3 ...]
"""
import asyncio
import logging
import sys
import time
from pathlib import Path

from cjm_substrate.core.manager import CapabilityManager
from cjm_substrate.core.queue import JobQueue
from cjm_transcription_core.cli import load_capabilities
from cjm_transcription_core.models import PipelineConfig
from cjm_transcription_core.pipeline import run_pipeline

SCRATCH_DIR = Path("/tmp/stage5_closeout_scratch")
SCRATCH_DB = str(SCRATCH_DIR / "context_graph.db")

SYSMON = "cjm-capability-monitor-nvidia"
FFMPEG = "cjm-capability-ffmpeg"
VAD = "cjm-capability-silero-vad"
TRANSCRIBERS = ["cjm-capability-whisper", "cjm-capability-voxtral-hf"]
GRAPH = "cjm-capability-graph-sqlite"
FFMPEG_CAP = 4  # the SG-33 raise under measurement


def lane_intervals(jobs, instance_id):
    """(start, end) UTC intervals for completed jobs of one instance."""
    return sorted((j.started_at, j.completed_at) for j in jobs
                  if j.capability_instance_id == instance_id and j.started_at and j.completed_at)


def merge(iv):
    """Merge overlapping intervals so a side can't double-count itself."""
    merged = []
    for s, e in sorted(iv):
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def overlap_seconds(intervals_a, intervals_b):
    """Total seconds where any interval of A overlaps any interval of B."""
    total = 0.0
    for sa, ea in merge(intervals_a):
        for sb, eb in merge(intervals_b):
            lo, hi = max(sa, sb), min(ea, eb)
            if hi > lo:
                total += (hi - lo).total_seconds()
    return total


def self_overlap_max(intervals):
    """Max simultaneously in-flight jobs within one lane + seconds at depth >= 2."""
    events = []
    for s, e in intervals:
        events.append((s, 1))
        events.append((e, -1))
    events.sort(key=lambda t: (t[0], -t[1]))
    depth = max_depth = 0
    over2 = 0.0
    prev = None
    for ts, d in events:
        if prev is not None and depth >= 2:
            over2 += (ts - prev).total_seconds()
        depth += d
        max_depth = max(max_depth, depth)
        prev = ts
    return max_depth, over2


async def main():
    # Bypassing cli.main() means configuring logging HERE — otherwise the
    # pipeline's logger.info lines (incl. verify results) are silently dropped.
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s")
    audio = [str(Path(p).resolve()) for p in sys.argv[1:]]
    if not audio:
        raise SystemExit("usage: measure_cold_parallel_overlap_e2e.py audio1.mp3 [audio2.mp3 ...]")
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    cfg = PipelineConfig(
        vad_capability=VAD,
        ffmpeg_capability=FFMPEG,
        transcriber_capabilities=TRANSCRIBERS,
        graph_capability=GRAPH,
        graph_db_path=SCRATCH_DB,
        max_segment_duration=300.0,
        sample_rate=16000,
        channels=1,
        force=False,          # fresh sources are cold by content; no cache to bypass
        assume_yes=True,
    )
    manager = CapabilityManager(search_paths=[Path(".cjm/manifests")], sysmon_capability_name=SYSMON)
    load_order = [SYSMON, FFMPEG, VAD] + TRANSCRIBERS + [GRAPH]
    load_capabilities(
        manager, load_order,
        configs={GRAPH: {"db_path": SCRATCH_DB}},
        max_concurrent={FFMPEG: FFMPEG_CAP},  # the --max-concurrent seam under test
    )
    queue = JobQueue(deps=manager, sysmon_capability_name=SYSMON)
    await queue.start()
    t0 = time.monotonic()
    try:
        manifest = await run_pipeline(manager, queue, cfg, audio)
        wall = time.monotonic() - t0
        jobs = list(queue._jobs.values())
        print(f"\n===== COLD-RUN PARALLEL-OVERLAP RESULTS =====")
        print(f"wall: {wall:.1f}s  jobs: {len(jobs)}")
        summed = sum((j.completed_at - j.started_at).total_seconds()
                     for j in jobs if j.started_at and j.completed_at)
        if summed:
            print(f"summed job durations: {summed:.1f}s  -> wall/summed ratio {wall/summed:.2f}")

        # transcriber co-run (different instances; admission-governed)
        a_iv = lane_intervals(jobs, TRANSCRIBERS[0])
        b_iv = lane_intervals(jobs, TRANSCRIBERS[1])
        co = overlap_seconds(a_iv, b_iv)
        a_sum = sum((e - s).total_seconds() for s, e in a_iv)
        b_sum = sum((e - s).total_seconds() for s, e in b_iv)
        print(f"\n{TRANSCRIBERS[0]} lane: {len(a_iv)} jobs, {a_sum:.1f}s busy")
        print(f"{TRANSCRIBERS[1]} lane: {len(b_iv)} jobs, {b_sum:.1f}s busy")
        if min(a_sum, b_sum):
            print(f"transcriber CO-RUN: {co:.1f}s ({100*co/min(a_sum, b_sum):.0f}% of the shorter lane)")

        # ffmpeg same-instance overlap under the raised cap
        f_iv = lane_intervals(jobs, FFMPEG)
        f_sum = sum((e - s).total_seconds() for s, e in f_iv)
        depth, over2 = self_overlap_max(f_iv)
        f_span = (max(e for _, e in f_iv) - min(s for s, _ in f_iv)).total_seconds() if f_iv else 0
        print(f"\nffmpeg lane (cap={FFMPEG_CAP}): {len(f_iv)} jobs, {f_sum:.1f}s busy, "
              f"span {f_span:.1f}s, MAX IN-FLIGHT {depth}, >=2-deep for {over2:.1f}s")

        for s in manifest.sources:
            print(f"\nemission [{Path(s.source_path).name}]: {s.graph}")
        out = Path("runs") / f"{manifest.run_id}.json"
        manifest.save(out)
        print(f"\nmanifest: {out}")
    finally:
        await queue.stop()
        for iid in reversed(load_order):
            try:
                manager.unload_capability(iid)
            except Exception as e:
                print(f"unload {iid} failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
