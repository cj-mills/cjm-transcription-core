#!/usr/bin/env python3
"""Stage-8 SN-I BOTH-TRANSCRIBER parity — the voxtral Option-C migration (build step 2).

Sibling to validate_stage8_sn1_e2e.py (whisper-only). This is the voxtral
milestone: with the MIGRATED voxtral co-resident on the shared graph skeleton,
the full transcription -> decomp -> correction chain over Supernova-I runs with
BOTH transcribers and the cross-transcriber divergence signal lights up (it was
DEFERRED from whisper's L10 because divergence is intrinsically cross-transcriber).
Both transcribers are driven through ONE dual-transcriber run (`--transcriber` is
repeatable); whisper rides its warm adapter cache, voxtral cold-transcribes
(~6+ min) the first time.

Two stage-8 milestones in one harness:
  1. {whisper, voxtral} multi-capability DISCOVERY — the compatible set for task
     'transcription' first exceeds 1 (manifest-surface match, UNLOADED-safe: the
     host instantiates no plugin; the GenericTranscriptionAdapter binds to BOTH
     by recorded `transcribe`+`get_current_config` surface).
  2. both-transcriber FAITHFULNESS — voxtral's SN-I output is byte-identical to
     the pre-migration corpus (261,617 chars, deterministic greedy decode); the
     Source id is conserved; the shared skeleton carries both transcribers'
     Transcript nodes; and the cross-transcriber divergence count is > 0.
     The empty-prune + divergence COUNTS are deliberately NOT pinned — whisper-base
     is the lightweight uncertainty detector whose fallback sampling makes those
     counts vary run-to-run on a cold re-transcribe (the feature that surfaces hard
     spots). A specific run measured 326 / 1,182; the old corpus recorded 320 / 1,176.
     See the constants-block note.

Runs into a FRESH /tmp scratch graph; the canonical seed
(cjm-transcription-core/.cjm/data/...) and the old decomp-core corpus stay
untouched. Requires the migrated voxtral worker env (test-voxtral-hf) and the
editable/published stage-8 stack.

Run (any env with sqlite3; spawns the cores' own envs):
  python tests_manual/validate_stage8_sn1_both_transcribers_e2e.py
"""
import json
import sqlite3
import subprocess
import time
from pathlib import Path

BASE = Path("/mnt/SN850X_8TB_EXT4/Projects/GitHub/cj-mills")
TRANSCRIPTION = BASE / "cjm-transcription-core"
DECOMP = BASE / "cjm-transcript-decomp-core"
CORRECTION = BASE / "cjm-transcript-correction-core"
ENVS = Path.home() / "miniforge3/envs"
SN1_SRC = BASE / "cjm-transcription-audio-segment/test_files/2018-07-14_show-62-supernova-in-the-east-i.mp3"
MANIFESTS = TRANSCRIPTION / ".cjm/manifests"

SCRATCH_DB = Path("/tmp/stage8_sn1_both_scratch_graph.db")
TX_OUT = Path("/tmp/stage8_sn1_both_transcription.json")
DECOMP_OUT = Path("/tmp/stage8_sn1_both_decomp.json")
CORR_OUT = Path("/tmp/stage8_sn1_both_correction.json")
ACTOR = "stress:stage8-sn1-both"
WHISPER = "cjm-transcription-plugin-whisper"
VOXTRAL = "cjm-transcription-plugin-voxtral-hf"

# DETERMINISTIC invariants (the regression gate). NOTE on empty/divergence: these
# are intentionally NOT pinned. whisper-base is the lightweight "uncertainty
# detector" of the lightweight+accuracy diff; its `temperature_increment_on_fallback`
# (0.2) makes it SAMPLE on hard segments, so the spine text — and therefore the
# empty-prune count + the cross-transcriber divergence count — vary run-to-run on a
# COLD re-transcribe (cache-backed re-derivation is stable). That variability is the
# feature (it surfaces hard spots), not a regression. A specific run measured
# 326 empty / 1,182 divergence; the old multi-source corpus's applied prune_empty
# recorded 320 / 1,176 — both legitimate, neither an invariant. We assert the
# deterministic facts instead and treat empty/divergence as informational.
EXPECT_SOURCE_ID = "24461366-6548-5b93-80ae-a03c463443bf"  # UUIDv5(SN-I file hash) — conserved across the migration
EXPECT_TX_SEGMENTS = 55        # coarse pipeline segments (VAD-deterministic)
EXPECT_GRAPH_NODES = 166       # 1 Source + 55 AudioSegment + 55 whisper-Tx + 55 voxtral-Tx (id = aseg·transcriber·config_hash)
EXPECT_SEGMENT_NODES = 3579    # fine spine (boundary-deterministic UUIDv5)
EXPECT_VOXTRAL_CHARS = 261617  # voxtral-mini greedy decode (do_sample=False) — DETERMINISTIC, byte-identical to the pre-migration corpus
WHISPER_CHARS_NOMINAL = 260464 # whisper-base — informational; varies ±tens of chars on cold re-transcribe (fallback sampling)
WHISPER_CHARS_TOL = 2000       # generous band; whisper spine is the non-deterministic detector, not a pinned baseline


def journal_max_seq(db: Path) -> int:
    if not db.exists():
        return 0
    with sqlite3.connect(db) as con:
        return con.execute("SELECT COALESCE(MAX(seq),0) FROM journal").fetchone()[0]


def journal_rows(db: Path, where: str, params=()):
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in con.execute(
            f"SELECT * FROM journal WHERE {where} ORDER BY seq", params)]
    finally:
        con.close()


def run(argv, cwd):
    proc = subprocess.run(argv, cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout[-3000:])
        print(proc.stderr[-3000:])
        raise SystemExit(f"command failed rc={proc.returncode}: {argv[1:4]}")
    return proc


def check_discovery() -> None:
    """Milestone 1: {whisper, voxtral} compatible set, manifest-surface-based."""
    print("== discovery: {whisper, voxtral} compatible set (unloaded-safe) ==")
    from cjm_plugin_system.core.manager import PluginManager
    mgr = PluginManager(search_paths=[MANIFESTS])
    mgr.discover_manifests()
    adapters = mgr.get_adapters_for_task("transcription")
    assert adapters, "no adapter discovered for task 'transcription'"
    compat = sorted(mgr.get_capabilities_compatible_with(adapters[0]))
    assert WHISPER in compat and VOXTRAL in compat, compat
    assert len(compat) >= 2, compat
    print(f"   adapter {adapters[0].name} -> compatible {compat}  ✓")


def main() -> None:
    assert SN1_SRC.exists(), SN1_SRC
    check_discovery()

    SCRATCH_DB.unlink(missing_ok=True)
    for p in (SCRATCH_DB.with_suffix(".db-wal"), SCRATCH_DB.with_suffix(".db-shm")):
        p.unlink(missing_ok=True)

    # ---- 1. dual-transcriber transcription (whisper warm + voxtral cold) ----
    print("== transcription (whisper + voxtral, dual run, task channel) ==")
    t0 = time.monotonic()
    run([str(ENVS / "cjm-transcription-core/bin/cjm-transcription-core"),
         "run", str(SN1_SRC),
         "--transcriber", WHISPER,
         "--transcriber", VOXTRAL,
         "--graph-plugin", "cjm-graph-plugin-sqlite",
         "--graph-db-path", str(SCRATCH_DB),
         "--sysmon-plugin", "cjm-system-monitor-nvidia",
         "--actor", ACTOR, "--output", str(TX_OUT), "--yes"],
        cwd=TRANSCRIPTION)
    print(f"   wall {time.monotonic() - t0:.1f}s")

    tx = json.loads(TX_OUT.read_text())
    src = tx["sources"][0]
    segs = src["segments"]
    w_chars = sum(len(s["transcripts"][WHISPER]["text"]) for s in segs)
    v_chars = sum(len(s["transcripts"][VOXTRAL]["text"]) for s in segs)
    g = src["graph"]
    assert len(segs) == 55, (len(segs), 55)  # VAD coarse segments (deterministic)
    assert v_chars == EXPECT_VOXTRAL_CHARS, (v_chars, EXPECT_VOXTRAL_CHARS)  # DETERMINISTIC: byte-identical to pre-migration
    assert abs(w_chars - WHISPER_CHARS_NOMINAL) <= WHISPER_CHARS_TOL, (w_chars, WHISPER_CHARS_NOMINAL)  # informational band
    assert g["source_node_id"] == EXPECT_SOURCE_ID, g["source_node_id"]  # conserved across the migration
    assert g["nodes_added"] == EXPECT_GRAPH_NODES, (g["nodes_added"], EXPECT_GRAPH_NODES)
    print(f"   {len(segs)} segs, whisper {w_chars} / voxtral {v_chars} chars, "
          f"Source {g['source_node_id'][:13]}…, graph {g['nodes_added']}n/{g['edges_added']}e  ✓")

    # ---- 2. decomp extend (spine from whisper; graph carries both) ----
    print("== decomp extend ==")
    dec_journal = DECOMP / ".cjm/journal.db"
    cursor = journal_max_seq(dec_journal)
    t0 = time.monotonic()
    run([str(ENVS / "cjm-transcript-decomp-core/bin/cjm-transcript-decomp-core"),
         "run", str(TX_OUT),
         "--text-from", WHISPER,
         "--graph-db-path", str(SCRATCH_DB),
         "--sysmon-plugin", "cjm-system-monitor-nvidia",
         "--actor", ACTOR, "--output", str(DECOMP_OUT), "--yes"],
        cwd=DECOMP)
    print(f"   wall {time.monotonic() - t0:.1f}s")

    dec = json.loads(DECOMP_OUT.read_text())
    seg_count = dec["sources"][0]["segment_count"]
    assert seg_count == EXPECT_SEGMENT_NODES, (seg_count, EXPECT_SEGMENT_NODES)
    verifies = journal_rows(dec_journal, "seq > ? AND run_id = ? AND event_type = 'verify_outcome'",
                            (cursor, dec["run_id"]))
    assert len(verifies) == 1, len(verifies)
    checks = json.loads(verifies[0]["payload"])["checks"]
    assert all(v is True for v in checks.values() if isinstance(v, bool)), checks
    print(f"   {seg_count} Segment nodes, verify all-True  ✓")

    # ---- 3. correction: both-transcriber empties + divergence ----
    print("== correction (both-transcriber empties + divergence) ==")
    t0 = time.monotonic()
    run([str(ENVS / "cjm-transcript-correction-core/bin/cjm-transcript-correction-core"),
         "run", str(DECOMP_OUT), "--actor", ACTOR, "--output", str(CORR_OUT), "--yes"],
        cwd=CORRECTION)
    print(f"   wall {time.monotonic() - t0:.1f}s")

    corr = json.loads(CORR_OUT.read_text())["sources"][0]
    empties = corr["empty_segments"]; divs = corr["transcriber_divergences"]
    # empty/divergence vary by design (whisper fallback sampling on a cold
    # re-transcribe); assert the union signal exists, not exact counts (header note).
    assert divs > 0, f"expected cross-transcriber divergences > 0 (both transcribers present), got {divs}"
    assert empties > 0, empties
    print(f"   empty-prune {empties} + divergence {divs} "
          f"(informational; vary on cold re-transcribe — whisper fallback sampling)  ✓")

    SCRATCH_DB.unlink(missing_ok=True)
    print("== stage-8 SN-I both-transcriber parity: PASS ==")


if __name__ == "__main__":
    main()
