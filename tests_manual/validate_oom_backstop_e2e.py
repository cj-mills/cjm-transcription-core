"""OOM backstop stress test (stage-3 ledger): Voxtral-Small-24B on a 24GB GPU.

Walks the full failure arc the stage-3 admission design leans on:
guaranteed-OOM model load on CUDA -> cuda_oom_to_plugin_resource_error ->
typed PluginResourceError over the wire error channel -> CR-7 evict + reload +
retry -> structured JobError on the failed member job (+ RETRY_STARTED event)
-> composition failure propagation (upstream convert node survives, run lands
failed) -> reconfigure device=cpu -> slow CPU success -> the empirical store
holds BOTH config rows (failed-on-cuda + succeeded-on-cpu), demonstrating the
config-hash keying that derived admission rests on.

Run from the repo root in the cjm-transcription-core env:
    python tests_manual/validate_oom_backstop_e2e.py
"""
import asyncio
import json
import logging
import os
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / ".cjm" / "manifests"
EMPIRICAL_DB = ROOT / ".cjm" / "empirical_resources.db"
AUDIO = ROOT / "test_files" / "short_test_audio.mp3"
MODEL = "mistralai/Voxtral-Small-24B-2507"
PLUGIN = "cjm-capability-voxtral-hf"
FFMPEG = "cjm-capability-ffmpeg"
SYSMON = "cjm-capability-monitor-nvidia"
PHASE1_ONLY = os.environ.get("OOM_PHASE1_ONLY") == "1"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s")
log = logging.getLogger("oom-stress")

from cjm_substrate.core.manager import CapabilityManager
from cjm_substrate.core.queue import JobQueue, JobStatus, JobEventType
from cjm_substrate.core.ports import (
    Composition, CompositionNode, NodeState, OutputRef,
)
from cjm_transcription_adapter_interface.core import TranscriptionResult  # noqa: F401 — wire-kind registration


def empirical_rows(instance_id: str) -> list:
    if not EMPIRICAL_DB.exists():
        return []
    con = sqlite3.connect(EMPIRICAL_DB)
    try:
        return con.execute(
            "SELECT config_hash, sample_count, success_count, gpu_memory_mb_peak_max,"
            " memory_mb_peak_max FROM empirical_resources WHERE instance_id=?",
            (instance_id,)).fetchall()
    finally:
        con.close()


def pipe_composition() -> Composition:
    return Composition(nodes=[
        CompositionNode("convert", FFMPEG, {
            "action": "convert", "input_path": str(AUDIO),
            "output_format": "wav", "sample_rate": 16000, "channels": 1,
        }),
        CompositionNode("transcribe", PLUGIN,
                        {"audio": OutputRef("convert", "output_path")}),
    ])


async def main() -> None:
    pm = CapabilityManager(search_paths=[MANIFESTS], sysmon_capability_name=SYSMON)
    pm.discover_manifests()

    def meta(name):
        return next(m for m in pm.discovered if m.name == name)

    # local_files_only: the shards are pre-downloaded; the plugin's full-repo
    # snapshot would otherwise re-pull the 48.5GB consolidated.safetensors that
    # from_pretrained never reads (sharded index load) — ledger G6 friction.
    pm.load_capability(meta(SYSMON))
    pm.load_capability(meta(FFMPEG))
    assert pm.load_capability(meta(PLUGIN), config={"model_id": MODEL, "device": "cuda",
                                            "local_files_only": True}), \
        f"failed to load {PLUGIN}"
    log.info(f"{PLUGIN} loaded with model_id={MODEL} device=cuda (guaranteed-OOM config)")

    queue = JobQueue(deps=pm, sysmon_capability_name=SYSMON)
    await queue.start()

    retry_events = []

    async def collect():
        async for evt in queue.all_events():
            if evt.type == JobEventType.RETRY_STARTED:
                retry_events.append(evt)
                log.info(f"RETRY_STARTED observed: attempt={evt.payload.get('attempt')} "
                         f"category={evt.payload.get('exception_category')}")

    collector = asyncio.create_task(collect())

    try:
        # ---- Phase 1: guaranteed OOM on GPU; the backstop arc must resolve
        # to a STRUCTURED failure, not a hang or a crash of the host.
        log.info("PHASE 1: submitting convert→transcribe composition (24B on cuda)...")
        t0 = time.time()
        comp_id = await queue.submit_composition(pipe_composition())
        run = await queue.wait_for_composition(comp_id)
        dt = time.time() - t0
        log.info(f"phase-1 composition resolved in {dt:.1f}s: status={run.status.value}")

        assert run.status == NodeState.failed, f"expected failed, got {run.status}"
        conv = run.node_runs["convert"]
        tr = run.node_runs["transcribe"]
        assert conv.state == NodeState.completed, \
            f"upstream convert should survive: {conv.state}"
        assert tr.state == NodeState.failed and tr.error is not None, \
            f"transcribe node should carry a structured error: {tr.state} {tr.error}"
        log.info(f"transcribe JobError: category={tr.error.category!r} "
                 f"retriable={tr.error.retriable} message={tr.error.message[:160]!r}")
        # G7: the typed unary error channel must deliver the RESOURCE category
        # to the host (pre-fix it collapsed to RuntimeError -> 'fatal').
        assert tr.error.category == "resource", \
            f"expected category 'resource' via the typed channel, got {tr.error.category!r}"
        log.info(f"RETRY_STARTED events: {len(retry_events)} (CR-7 fired)")
        # G7: CR-7's reactive retry must actually engage on the unary path.
        assert len(retry_events) >= 1, \
            "CR-7 reactive retry did not fire (RETRY_STARTED count 0)"

        rows = empirical_rows(PLUGIN)
        log.info(f"empirical rows after phase 1: {rows}")
        if PHASE1_ONLY:
            failed_rows = [r for r in rows if r[2] < r[1]]
            assert failed_rows, f"expected a failed-config row: {rows}"
            log.info("OOM BACKSTOP STRESS (phase 1 only): ALL CHECKS PASSED")
            return

        # ---- Phase 2: reconfigure device=cpu; the SAME composition succeeds.
        log.info("PHASE 2: reconfiguring device=cpu (model reload to system RAM)...")
        pm.update_plugin_config(PLUGIN, {"model_id": MODEL, "device": "cpu",
                                     "local_files_only": True})
        t1 = time.time()
        comp_id2 = await queue.submit_composition(pipe_composition())
        run2 = await queue.wait_for_composition(comp_id2)
        dt2 = time.time() - t1
        log.info(f"phase-2 composition resolved in {dt2:.1f}s: status={run2.status.value}")
        assert run2.status == NodeState.completed, \
            f"CPU run should succeed: {run2.status}; " \
            f"transcribe={run2.node_runs['transcribe'].error}"
        result = run2.results_by_node()["transcribe"]
        text = str(result.text or "")
        log.info(f"CPU transcription ({len(text)} chars): {text[:120]!r}")
        assert text.strip(), "empty CPU transcription"

        # ---- The two-row evidence: same instance, two config hashes —
        # the keying that makes derived admission self-correcting.
        rows = empirical_rows(PLUGIN)
        log.info(f"empirical rows after phase 2: {rows}")
        assert len(rows) >= 2, f"expected cuda + cpu config rows, got {rows}"
        failed_rows = [r for r in rows if r[2] < r[1]]
        assert failed_rows, f"expected a row with success_count < sample_count: {rows}"
        log.info("OOM BACKSTOP STRESS: ALL CHECKS PASSED")
    finally:
        collector.cancel()
        await queue.stop()
        for name in (PLUGIN, FFMPEG, SYSMON):
            try:
                pm.unload_capability(name)
            except Exception:
                pass


if __name__ == "__main__":
    asyncio.run(main())
    sys.exit(0)
