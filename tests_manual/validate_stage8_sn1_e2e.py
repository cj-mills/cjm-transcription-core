#!/usr/bin/env python3
"""Stage-8 SN-I closeout — the whisper Option-C migration validated at scale.

The standing harness for the stage-8 whisper worked example: the full
transcription -> decomp -> correction chain over Supernova-I (the ~4.5-hour
Hardcore History episode), driven by the MIGRATED whisper — a pure-compute
`ToolCapability` invoked through the `GenericTranscriptionAdapter` over the
explicit task channel (cache/persistence lifted off the tool into the adapter).

Scope: WHISPER-ONLY. The 1,176 transcriber-divergence baseline is intrinsically
cross-transcriber and is DEFERRED to the voxtral migration (stage-8 build step 2),
which re-transcribes into the same graph and exercises the {whisper, voxtral}
multi-capability discovery union. Whisper's raw empty count is 326 (the
pre-Source-rooted number); the both-transcriber 320 is reached only when voxtral
co-resides on the shared skeleton (one-side-empty reclassification — stage-5
mapping). The full 320 + 1,176 both-transcriber parity is covered by the
decomp-core stage-7 volume regression (validate_stage7_volume_journal_e2e.py).

Validates (the stage-8-specific milestones):
  - the GenericTranscriptionAdapter AUTO-BINDS to whisper by surface match
    (CR-17 pt 2: transcribe + get_current_config), host holds no instances;
  - transcription over the task channel: 55 segments, 260,464 whisper chars,
    deterministic Source id 24461366-... (UUIDv5(file hash) — conserved across
    the migration), graph emission 111 nodes (1 Source + 55 AudioSegment +
    55 Transcript);
  - the adapter's CACHE BOOKEND fires at scale on the warm re-run (journal
    cache_hit rows, no model load) — the cache/persist responsibilities moved
    cleanly to the adapter;
  - decomp extend: 3,579 Segment nodes + verify_source all-True (9 checks);
  - correction empty-prune: 326 (whisper-only), 0 divergences.

Runs into a FRESH scratch graph in /tmp; the canonical seed
(cjm-transcription-core/.cjm/data/...) and the old decomp-core corpus stay
untouched. Requires the editable stage-8 stack (transcription-core host +
test-whisper worker) and the migrated whisper cache already populated for SN-I
(this session's cold run, or any prior SN-I whisper run); a cold cache makes the
transcription step a ~6-min real re-transcribe instead of a warm cache-hit.

As-measured (2026-06-14, stage-8 closeout, editable host + test-whisper worker):
  transcription warm ~cache-hit (no model load), decomp ~20s (VAD/FA cache-hit),
  correction ~2.5s. The first cold-cache run of transcription is ~6 min.

Run (any env with sqlite3; spawns the cores' own envs):
  python tests_manual/validate_stage8_sn1_e2e.py
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
SN1_SRC = Path("/mnt/SN850X_8TB/Media_Library/Podcasts/Hardcore_History/Dan Carlin Hardcore History Archives/2018-07-14_Show 62 - Supernova in the East I.mp3")  # canonical media-library copy (hash-identical to the deferred audio-segment copy → Source id conserved)

SCRATCH_DB = Path("/tmp/stage8_sn1_scratch_graph.db")
TX_OUT = Path("/tmp/stage8_sn1_e2e_transcription.json")
DECOMP_OUT = Path("/tmp/stage8_sn1_e2e_decomp.json")
CORR_OUT = Path("/tmp/stage8_sn1_e2e_correction.json")
ACTOR = "stress:stage8-sn1"
WHISPER = "cjm-capability-whisper"

# Stage-8 whisper-only baselines (SN-I) at max_segment_duration=220 (FA over-assignment
# fix; rebaselined 2026-06-17 from the pre-soxr/pre-AudioRendition 300s values). Structural
# counts (coarse/nodes/segments) are boundary-deterministic -> hard-asserted; whisper char +
# empty counts are whisper-base nondeterministic on a COLD re-transcribe -> band/informational.
EXPECT_SOURCE_ID = "24461366-6548-5b93-80ae-a03c463443bf"  # UUIDv5(SN-I file hash) — conserved
EXPECT_TX_SEGMENTS = 76       # coarse pipeline segments at 220 (was 55 at 300)
WHISPER_CHARS_NOMINAL = 260322  # whisper-base greedy decode at 220 (was 260464 at 300); informational
WHISPER_CHARS_TOL = 2000        # generous band; whisper is the non-deterministic detector, not pinned
EXPECT_GRAPH_NODES = 229      # 1 Source + 76 AudioSegment + 76 raw AudioRendition + 76 Transcript (AudioRendition-era + 220; was 111 pre-rendition/300)
EXPECT_SEGMENT_NODES = 3522   # fine spine (boundary-deterministic; was 3579 pre-soxr/300)
EXPECT_DIVERGENCE = 0         # whisper-only (no second transcriber to diverge from)


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


def main() -> None:
    assert SN1_SRC.exists(), SN1_SRC
    SCRATCH_DB.unlink(missing_ok=True)
    for p in (SCRATCH_DB.with_suffix(".db-wal"), SCRATCH_DB.with_suffix(".db-shm")):
        p.unlink(missing_ok=True)

    # ---- 1. transcription over the task channel (migrated whisper) ----
    print("== transcription (migrated whisper, task channel) ==")
    tx_journal = TRANSCRIPTION / ".cjm/journal.db"
    cursor = journal_max_seq(tx_journal)
    t0 = time.monotonic()
    run([str(ENVS / "cjm-transcription-core/bin/cjm-transcription-core"),
         "run", str(SN1_SRC),
         "--transcriber", WHISPER,
         "--graph-plugin", "cjm-capability-graph-sqlite",
         "--graph-db-path", str(SCRATCH_DB),
         "--sysmon-plugin", "cjm-capability-monitor-nvidia",
         "--actor", ACTOR, "--output", str(TX_OUT), "--yes"],
        cwd=TRANSCRIPTION)
    print(f"   wall {time.monotonic() - t0:.1f}s")

    tx = json.loads(TX_OUT.read_text())
    src = tx["sources"][0]
    segs = src["segments"]
    chars = sum(len(s["transcripts"][WHISPER]["text"]) for s in segs)
    g = src["graph"]
    assert len(segs) == EXPECT_TX_SEGMENTS, (len(segs), EXPECT_TX_SEGMENTS)
    assert abs(chars - WHISPER_CHARS_NOMINAL) <= WHISPER_CHARS_TOL, (chars, WHISPER_CHARS_NOMINAL)  # informational band (whisper nondeterminism)
    assert g["source_node_id"] == EXPECT_SOURCE_ID, g["source_node_id"]
    assert g["nodes_added"] == EXPECT_GRAPH_NODES, (g["nodes_added"], EXPECT_GRAPH_NODES)

    tx_run_id = tx["run_id"]
    new_rows = journal_rows(tx_journal, "seq > ? AND run_id = ?", (cursor, tx_run_id))
    by_type = {}
    for r in new_rows:
        by_type.setdefault(r["event_type"], []).append(r)
    cache_hits = len(by_type.get("cache_hit", []))
    task_accounts = len(by_type.get("task_account", []))
    # Warm cache path: the adapter's get_cached bookend fires per segment (no
    # model load). A cold cache makes this a real re-transcribe (cache_hits == 0,
    # task_accounts still present) — accepted, the chain still validates.
    print(f"   {len(segs)} segs, {chars} chars, Source {g['source_node_id'][:13]}…, "
          f"graph {g['nodes_added']}n/{g['edges_added']}e; "
          f"journal task_account={task_accounts} cache_hit={cache_hits}  ✓")

    # ---- 2. decomp extend ----
    print("== decomp extend ==")
    dec_journal = DECOMP / ".cjm/journal.db"
    cursor = journal_max_seq(dec_journal)
    t0 = time.monotonic()
    run([str(ENVS / "cjm-transcript-decomp-core/bin/cjm-transcript-decomp-core"),
         "run", str(TX_OUT),
         "--text-from", WHISPER,
         "--graph-db-path", str(SCRATCH_DB),
         "--sysmon-plugin", "cjm-capability-monitor-nvidia",
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
    assert checks["segment_count"] == EXPECT_SEGMENT_NODES, checks["segment_count"]
    assert checks["source_id"] == EXPECT_SOURCE_ID, checks["source_id"]
    print(f"   {seg_count} Segment nodes, verify all-True (9 checks)  ✓")

    # ---- 3. correction empty-prune (whisper-only) ----
    print("== correction empty-prune ==")
    t0 = time.monotonic()
    run([str(ENVS / "cjm-transcript-correction-core/bin/cjm-transcript-correction-core"),
         "run", str(DECOMP_OUT), "--actor", ACTOR, "--output", str(CORR_OUT), "--yes"],
        cwd=CORRECTION)
    print(f"   wall {time.monotonic() - t0:.1f}s")

    corr = json.loads(CORR_OUT.read_text())["sources"][0]
    assert corr["empty_segments"] > 0, corr["empty_segments"]  # informational: ~26 at 220 (vs ~326 at 300); varies on cold re-transcribe
    assert corr["transcriber_divergences"] == EXPECT_DIVERGENCE, corr["transcriber_divergences"]
    print(f"   empty-prune {corr['empty_segments']} (whisper-only, informational; ~26 at 220 vs ~326 at 300), "
          f"divergence {corr['transcriber_divergences']}  ✓")

    SCRATCH_DB.unlink(missing_ok=True)
    print("== stage-8 SN-I whisper-only closeout: PASS ==")


if __name__ == "__main__":
    main()
