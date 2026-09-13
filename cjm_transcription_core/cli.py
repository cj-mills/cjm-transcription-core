"""The CLI driver — the workflow core's first (and currently only) frontend.

Ships in-package as the `cjm-transcription-core` console script so the driver can
never skew from the core. GUI presentation drivers come later and consume the same
`pipeline` module; they never reimplement it (CLI-first / headless-core principle).

Prerequisite runtime (once, from the repo root):

    cjm-ctl --cjm-config cjm.yaml setup-runtime
    cjm-ctl --cjm-config cjm.yaml install-all --capabilities capabilities_test.yaml --force

Then e.g.:

    cjm-transcription-core run path/to/audio.mp3 --yes
    cjm-transcription-core run ep1.mp3 ep2.mp3 --transcriber cjm-capability-voxtral-hf
    # GPU scale runs: opt into CR-7 GPU subtree attribution (records gpu_memory_mb_peak)
    cjm-transcription-core run ep1.mp3 --yes --sysmon-capability cjm-capability-monitor-nvidia
    # Stage 5: dual-transcriber (lightweight + accuracy) run WITH graph-root emission
    # (Source -> AudioSegment -> Transcript; idempotent under cache hits)
    cjm-transcription-core run ep1.mp3 --yes \\
      --transcriber cjm-capability-whisper \\
      --transcriber cjm-capability-voxtral-hf \\
      --graph-capability cjm-capability-graph-sqlite --sysmon-capability cjm-capability-monitor-nvidia
"""

import argparse
import asyncio
import getpass
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from cjm_context_graph_layer.journal import sidecar_journal_path
from cjm_substrate.core.journal_store import SubstrateEventType
from cjm_substrate.core.manager import CapabilityManager
from cjm_substrate.core.queue import JobQueue
from cjm_substrate.core.workspace import resolve_workspace
from cjm_transcript_graph_schema.schema import external_config_hash, external_transcriber_name
from cjm_transcription_core.chunk import (apply_chunk_update, census_rows, chunks_from_census,
                                          derive_manifest, fetch_transcript_rows,
                                          land_chunk_transcript, load_run_manifest,
                                          prior_config_hash, PRODUCER_EXTERNAL, PRODUCER_RERUN,
                                          prompt_hash_of, save_manifest, select_chunks,
                                          summarize_census)
from cjm_transcription_core.curation import (add_reference, declare_structure, retract_reference,
                                             structure_entries_from_map)
from cjm_transcription_core.models import CollectionDecl, new_run_id, PipelineConfig
from cjm_transcription_core.pipeline import (_journal_run_event, collect_capability_info,
                                             run_pipeline, submit_and_wait)

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:  # Configured CLI parser
    """Build the CLI parser (subcommands: run)."""
    parser = argparse.ArgumentParser(
        prog="cjm-transcription-core",
        description="Headless transcription pipeline: VAD -> segment -> convert -> transcribe [-> graph emission].",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run the pipeline over one or more audio files")
    run.add_argument("audio", nargs="+",
                     help="Source audio/video file path(s) and/or directories, in order; "
                          "a directory expands to every media file under it (recursive, sorted)")
    run.add_argument("--manifests-dir", default=".cjm/manifests", help="Capability manifests directory")
    run.add_argument("--transcriber", action="append", default=None,
                     help="Transcriber spec NAME[@INSTANCE_ID][:key=value,...]; REPEATABLE for the "
                          "dual-transcriber (lightweight + accuracy) comparison run. @INSTANCE_ID + "
                          "config overrides stand up several (capability, MODEL) instances of ONE "
                          "capability side by side (e.g. cjm-capability-whisper@whisper-tiny:model=tiny) "
                          "(default: cjm-capability-whisper)")
    run.add_argument("--vad-capability", default="cjm-capability-silero-vad", help="VAD capability name")
    run.add_argument("--ffmpeg-capability", default="cjm-capability-ffmpeg", help="Convert/segment capability name")
    run.add_argument("--preprocessing-capability", default=None,
                     help="Opt-in audio-preprocessing capability (e.g. cjm-capability-demucs for vocals "
                          "isolation); runs per-segment on FULL-BAND audio BEFORE the model-input convert, "
                          "via the source_separation task channel (default: no preprocessing)")
    run.add_argument("--diarization-capability", default=None,
                     help="Speaker-diarization capability (default: cjm-capability-pyannote when "
                          "installed — the untouched default degrades to a warning if missing; an "
                          "explicit name fails loudly). Default-ON: runs ONCE per source on the "
                          "full decoded PCM rendition; turns persist SOURCE-KEYED under "
                          "<workspace>/diarization/ so existing spines inherit them")
    run.add_argument("--no-diarization", action="store_true",
                     help="Disable the default-on speaker-diarization rung")
    run.add_argument("--graph-capability", default=None,
                     help="Graph-storage capability for Source/AudioSegment/Transcript emission "
                          "(CR-18 revolution 2); default: no emission, manifest-only run")
    run.add_argument("--graph-db-path", default=None,
                     help="Explicit graph DB path override (caller-wins config, C8/F10; default: the capability's configured db_path)")
    run.add_argument("--sysmon-capability", default=None, help="monitor capability for GPU subtree attribution (CR-7); loaded first; default: no monitor")
    run.add_argument("--max-concurrent", action="append", default=None, metavar="NAME=N",
                     help="Per-capability SG-33 max_concurrent_requests override, REPEATABLE "
                          "(e.g. --max-concurrent cjm-capability-ffmpeg=4); same-worker "
                          "concurrency is opt-in — subprocess-backed workers parallelize, "
                          "model workers stay serial-per-instance (default: unset = 1)")
    run.add_argument("--max-segment-duration", type=float, default=220.0, help="Wall-clock cap per segment in seconds (220 keeps each forced-alignment input clear of the qwen3-FA ~240-250s degeneracy cliff; FA over-assignment investigation 2026-06-16)")
    run.add_argument("--sample-rate", type=int, default=16000, help="Model-input sample rate")
    run.add_argument("--channels", type=int, default=1, help="Model-input channel count")
    run.add_argument("--force", action="store_true", help="Bypass capability-side caches (VAD + transcription + preprocessing)")
    run.add_argument("-y", "--yes", action="store_true", help="Auto-accept HITL seams (headless mode)")
    run.add_argument("--output", default=None, help="Run-manifest output path (default: <workspace>/runs/<run_id>.json when a workspace is active, else runs/<run_id>.json under the cwd)")
    run.add_argument("--workspace", default=None,
                     help="Workspace root (5daadfc4; default: CJM_WORKSPACE env, else upward walk "
                          "from cwd). Supplies the runs/ output default and is exported so "
                          "substrate config + capability workers resolve workspace-scoped paths")
    run.add_argument("--actor", default=None,
                     help="Journal attribution for who/what initiated this run (default: cli:<username>)")
    run.add_argument("--collection", default=None, metavar="TITLE",
                     help="File ALL of this run's sources into the named collection (a human "
                          "naming it lands status=confirmed; replaces the automatic per-folder "
                          "proposals; ae3464fc)")
    run.add_argument("--no-collection", action="store_true",
                     help="Suppress the automatic folder->collection proposal (a directory arg "
                          "otherwise proposes a collection named after the folder)")
    run.add_argument("-v", "--verbose", action="store_true", help="DEBUG-level logging")

    # ---- declare-structure: the source structure map (DEC 8d9de793 clause 7) ----
    decl = sub.add_parser(
        "declare-structure",
        help="Land a human-confirmed SOURCE STRUCTURE MAP (a work's part/chapter cells per "
             "member Source) as journaled work_structure property merges")
    decl.add_argument("--map", required=True,
                      help="Structure-map JSON: {collection_id?, entries: [{source_id, evidence?, "
                           "...cells}]} — cells are the work's own structure (kind / part / "
                           "part_title / chapter / unit / title), evidence says where each came from")
    decl.add_argument("--collection-id", default=None,
                      help="Collection node id the map lies over (default: the document's)")
    decl.add_argument("--manifests-dir", default=".cjm/manifests", help="Capability manifests directory")
    decl.add_argument("--graph-capability", default="cjm-capability-graph-sqlite",
                      help="Graph-storage capability name")
    decl.add_argument("--graph-db-path", default=None,
                      help="Graph db path (default: the workspace capability config)")
    decl.add_argument("--workspace", default=None,
                      help="Workspace root (default: CJM_WORKSPACE env, else upward walk from cwd)")
    decl.add_argument("--actor", default=None,
                      help="Attribution (default: human:<user> — the map is a human confirmation)")
    decl.add_argument("--dry-run", action="store_true",
                      help="Parse + print the entries; touch neither graph nor journal")
    decl.add_argument("-v", "--verbose", action="store_true", help="DEBUG-level logging")

    # ---- add-reference / retract-reference: human-added resource links on a Source (ae103970) ----
    def _graph_plumbing(p: argparse.ArgumentParser) -> None:  # the declare-structure graph-stack flags, shared
        p.add_argument("--manifests-dir", default=".cjm/manifests", help="Capability manifests directory")
        p.add_argument("--graph-capability", default="cjm-capability-graph-sqlite",
                       help="Graph-storage capability name")
        p.add_argument("--graph-db-path", default=None,
                       help="Graph db path (default: the workspace capability config)")
        p.add_argument("--workspace", default=None,
                       help="Workspace root (default: CJM_WORKSPACE env, else upward walk from cwd)")
        p.add_argument("--actor", default=None, help="Attribution (default: human:<user>)")
        p.add_argument("-v", "--verbose", action="store_true", help="DEBUG-level logging")
    addr = sub.add_parser(
        "add-reference",
        help="Attach a HUMAN-ADDED resource link to a Source as a Reference node (publisher page, "
             "author post, related notes, cited work) — journaled; every rendering of the unit carries it")
    addr.add_argument("--source-id", required=True, help="The Source node id (or unique prefix)")
    addr.add_argument("--label", required=True, help="Reader-facing link text")
    addr.add_argument("--url", default="", help="The public URL (the fallback target while --notes-slug is unborn)")
    addr.add_argument("--notes-slug", default="",
                      help="A notes-graph Note slug this link points at (cross-work link; resolves once born)")
    addr.add_argument("--role", default="related",
                      help="Open vocabulary; recommended: publisher-page | author-post | related-notes | cited-work")
    _graph_plumbing(addr)
    retr = sub.add_parser("retract-reference", help="Retract a Reference node (cascade deletes its edge) — journaled")
    retr.add_argument("reference_id", help="The Reference node id")
    _graph_plumbing(retr)
    # ---- retire-collection: a journaled FACT, never a cascade (ruling a7617bd4, item eaefebd2) ----
    rcol = sub.add_parser(
        "retire-collection",
        help="Retire a Collection as a journaled fact (status=retired): pickers and the runaway "
             "census hide it; its Sources, spines and corrections stay untouched. --unretire restores")
    rcol.add_argument("collection", help="Collection node id (prefix ok) or exact title")
    rcol.add_argument("--reason", default="", help="Why (journaled; e.g. re-downloaded as a fresh collection)")
    rcol.add_argument("--unretire", action="store_true", help="Restore a retired collection to confirmed")
    _graph_plumbing(rcol)

    # ---- chunk-grain re-runs, external landings, the runaway census (cf0b91d6) ----
    def _chunk_plumbing(p: argparse.ArgumentParser) -> None:  # shared by rerun-chunk / add-transcript
        p.add_argument("--manifest", required=True,
                       help="The PARENT transcription run manifest (runs/run_*.json); never patched — "
                            "the landing writes a DERIVED manifest beside it (ruling 910f3692)")
        p.add_argument("--source", default=None,
                       help="Source selector: an index, a content-hash prefix, or a source_path substring "
                            "(default: every source in the manifest)")
        p.add_argument("--segment", action="append", type=int, default=None, metavar="N",
                       help="Segment index within the selected source(s); REPEATABLE (default: all)")
        p.add_argument("--reason", default=None,
                       help="Why this landing happens (journaled; default: runaway for a re-run, escalation for a paste)")
        p.add_argument("--output", default=None,
                       help="Derived-manifest path (default: <runs dir of the parent>/<new run id>.json)")
        p.add_argument("--dry-run", action="store_true", help="Resolve the targets and print them; touch nothing")
        _graph_plumbing(p)
    rr = sub.add_parser(
        "rerun-chunk",
        help="Re-transcribe selected chunks (AudioSegments) of a past run with ONE transcriber, cache "
             "bypassed; each new Transcript variant lands under the existing AudioSegment and "
             "SUPERSEDES the prior variant; a derived manifest records the landings")
    rr.add_argument("--transcriber", required=True,
                    help="The manifest's transcriber key to re-run (e.g. cjm-capability-voxtral-hf or whisper--small); "
                         "it reloads with the manifest's recorded config for that instance")
    rr.add_argument("--transcriber-config", action="append", default=None, metavar="KEY=VALUE",
                    help="Config override on top of the recorded config, REPEATABLE (JSON-parsed values)")
    rr.add_argument("--flagged", action="store_true",
                    help="Target = the runaway census's flagged chunks for this transcriber within the "
                         "manifest's sources (intersected with --source/--segment when given)")
    rr.add_argument("--max-chars", type=int, default=20000, help="Census: oversized-text threshold")
    rr.add_argument("--max-words-per-second", type=float, default=8.0, help="Census: implausible speech rate")
    rr.add_argument("--disagreement-ratio", type=float, default=4.0, help="Census: two-transcriber word-count ratio")
    rr.add_argument("--fail-fast", action="store_true", help="Stop at the first failed chunk (default: record + continue)")
    rr.add_argument("--sysmon-capability", default=None, help="monitor capability for GPU attribution (loaded first)")
    _chunk_plumbing(rr)
    at = sub.add_parser(
        "add-transcript",
        help="Land an operator-pasted EXTERNAL transcript for ONE chunk as a third transcriber "
             "(<model id>/manual) with provenance — the same landing a re-run uses")
    at.add_argument("--model-id", required=True, help="The external model that produced the text (e.g. gemini-2.5-pro)")
    at.add_argument("--text-file", required=True, help="File holding the pasted transcript text ('-' = stdin)")
    at.add_argument("--prompt-file", default=None,
                    help="The prompt template used (hashed into the variant's config hash; the prompt is data)")
    at.add_argument("--prompt-hash", default=None, help="A precomputed prompt hash instead of --prompt-file")
    at.add_argument("--text-source", default="paste",
                    help="Where the text came from (e.g. 'gemini web ui'); recorded in the landing provenance")
    _chunk_plumbing(at)
    rc = sub.add_parser(
        "runaway-census",
        help="List the LIVE chunk variants that need a better transcription: degenerate-tail markers "
             "(new runs), oversized / implausible words-per-second text (old runs), extreme "
             "two-transcriber disagreement; superseded variants are not live")
    rc.add_argument("--collection", action="append", default=None, metavar="TITLE",
                    help="Restrict to a collection title; REPEATABLE (default: all)")
    rc.add_argument("--transcriber", default=None, help="Restrict the flagged rows to one transcriber")
    rc.add_argument("--max-chars", type=int, default=20000, help="Oversized-text threshold")
    rc.add_argument("--max-words-per-second", type=float, default=8.0, help="Implausible speech rate over the chunk")
    rc.add_argument("--disagreement-ratio", type=float, default=4.0, help="Two-transcriber word-count ratio that flags")
    rc.add_argument("--include-superseded", action="store_true", help="List superseded variants too (marked)")
    rc.add_argument("--include-escalated", action="store_true",
                    help="List chunks already covered by an external (/manual) transcript too (marked ESCALATED)")
    rc.add_argument("--include-retired", action="store_true",
                    help="Count sources of RETIRED collections too (default: hidden, ruling a7617bd4)")
    rc.add_argument("--json", default=None, metavar="PATH", help="Write the flagged rows + summary as JSON")
    rc.add_argument("--limit", type=int, default=50, help="Rows to print (the JSON carries all)")
    _graph_plumbing(rc)
    return parser


def parse_max_concurrent(
    values: Optional[List[str]],  # Repeatable NAME=N CLI values (None = no overrides)
) -> Dict[str, int]:  # Capability name -> SG-33 max_concurrent_requests
    """Parse repeatable `--max-concurrent NAME=N` values into a per-capability cap map."""
    out: Dict[str, int] = {}
    for v in values or []:
        name, sep, n = v.partition("=")
        if not sep or not name:
            raise SystemExit(f"--max-concurrent expects NAME=N, got {v!r}")
        try:
            cap = int(n)
        except ValueError:
            raise SystemExit(f"--max-concurrent expects an integer cap, got {v!r}")
        if cap < 1:
            raise SystemExit(f"--max-concurrent cap must be >= 1, got {v!r}")
        out[name] = cap
    return out


def load_capabilities(
    manager: CapabilityManager,   # Freshly constructed manager
    instance_ids: List[Any],  # Capability names (default instances) and/or parse_transcriber_spec load directives
    configs: Optional[Dict[str, Dict[str, Any]]] = None,  # Per-capability config overrides (caller-wins, C8)
    max_concurrent: Optional[Dict[str, int]] = None,  # Per-instance SG-33 max_concurrent_requests (unset = queue default of 1)
) -> None:
    """Discover manifests + load each requested capability.

    A plain-string item loads the DEFAULT instance (name = instance id, stage-5
    behavior). A dict directive ({"capability", "instance_id", "config"} — the
    parse_transcriber_spec shape) loads a CR-10 NAMED instance so one capability
    can host several (capability, MODEL) instances side by side (db200725).
    """
    manager.discover_manifests()
    discovered = {m.name: m for m in manager.discovered}
    for item in instance_ids:
        directive = item if isinstance(item, dict) else {"capability": item, "instance_id": item, "config": {}}
        name = directive["capability"]
        iid = directive["instance_id"]
        meta = discovered.get(name)
        if meta is None:
            raise SystemExit(
                f"capability {name!r} not found in manifests "
                f"(discovered: {sorted(discovered)}) — run cjm-ctl install-all first"
            )
        config = directive["config"] or (configs or {}).get(iid)
        if not manager.load_capability(meta, config=config,
                                   instance_id=(iid if iid != name else None),
                                   max_concurrent_requests=(max_concurrent or {}).get(iid)):
            raise SystemExit(f"failed to load capability {iid!r}")
        logger.info(f"loaded {iid}" + (f" ({name})" if iid != name else ""))


async def run_command(
    args: argparse.Namespace,  # Parsed CLI arguments for the `run` subcommand
) -> int:  # Process exit code (0 = all sources completed)
    """Execute the `run` subcommand: full pipeline over the given audio files."""
    # 5daadfc4 workspace: resolve BEFORE any substrate config loads. Exporting
    # CJM_WORKSPACE makes the whole process tree (substrate config, capability
    # workers via CJM_CAPABILITY_DATA_DIR injection) workspace-scoped; the flag
    # form keeps the printed hand-off command reproducible standalone.
    ws = resolve_workspace(explicit=getattr(args, "workspace", None))
    if ws is not None:
        os.environ["CJM_WORKSPACE"] = str(ws.root)
    specs = [parse_transcriber_spec(s) for s in (args.transcriber or ["cjm-capability-whisper"])]
    transcribers = [s["instance_id"] for s in specs]
    if len(set(transcribers)) != len(transcribers):
        raise SystemExit(f"duplicate transcriber instance ids: {transcribers}")
    cfg = PipelineConfig(
        vad_capability=args.vad_capability,
        ffmpeg_capability=args.ffmpeg_capability,
        transcriber_capabilities=transcribers,
        preprocessing_capability=args.preprocessing_capability,
        diarization_capability=(None if args.no_diarization
                                else (args.diarization_capability or "cjm-capability-pyannote")),
        diarization_root=(str(ws.root) if ws is not None else None),
        graph_capability=args.graph_capability,
        graph_db_path=args.graph_db_path,
        max_segment_duration=args.max_segment_duration,
        sample_rate=args.sample_rate,
        channels=args.channels,
        force=args.force,
        assume_yes=args.yes,
    )
    # CR-14 follow-up: actor attribution (operator identity by default; agents/
    # services pass --actor explicitly). Computed here so the collection
    # declarations carry the same attribution as the run.
    actor = args.actor or f"cli:{getpass.getuser()}"
    sources, collection_decls = expand_sources_with_collections(
        args.audio, explicit_title=args.collection,
        no_collection=args.no_collection, actor=actor)
    if args.graph_db_path and not args.graph_capability:
        raise SystemExit("--graph-db-path requires --graph-capability")
    max_concurrent = parse_max_concurrent(args.max_concurrent)

    # CR-7 GPU subtree attribution is opt-in: --sysmon-capability threads the monitor
    # name into BOTH the manager (load-time empirical records) and the queue
    # (per-job resource samples); the monitor loads FIRST so GPU capabilities'
    # samples record gpu_memory_mb_peak (voxtral-vllm e2e pattern).
    manager = CapabilityManager(
        search_paths=[Path(args.manifests_dir)],
        sysmon_capability_name=args.sysmon_capability,
    )
    # Default-ON diarization degrades gracefully: the UNTOUCHED default skips
    # with a warning when the capability isn't installed; an explicit
    # --diarization-capability still fails loudly in load_capabilities.
    if cfg.diarization_capability and not args.diarization_capability:
        manager.discover_manifests()
        if cfg.diarization_capability not in {m.name for m in manager.discovered}:
            logger.warning(f"default diarization capability {cfg.diarization_capability!r} "
                           "not installed — running without speaker diarization")
            cfg.diarization_capability = None
    # Preprocessing (opt-in) loads alongside the other compute capabilities; its
    # adapter auto-binds by surface match exactly like VAD/transcription.
    instance_ids = ([cfg.ffmpeg_capability, cfg.vad_capability]
                    + ([cfg.diarization_capability] if cfg.diarization_capability else [])
                    + ([cfg.preprocessing_capability] if cfg.preprocessing_capability else [])
                    + list(specs)
                    + ([cfg.graph_capability] if cfg.graph_capability else []))
    load_order = ([args.sysmon_capability] if args.sysmon_capability else []) + instance_ids
    # Teardown iterates INSTANCE IDS (a spec directive loads under its instance_id).
    loaded_ids = [i["instance_id"] if isinstance(i, dict) else i for i in load_order]
    # --graph-db-path threads a caller-wins config into the graph load (C8/F10).
    configs = ({cfg.graph_capability: {"db_path": args.graph_db_path}}
               if (cfg.graph_capability and args.graph_db_path) else None)
    load_capabilities(manager, load_order, configs=configs, max_concurrent=max_concurrent)

    queue = JobQueue(deps=manager, sysmon_capability_name=args.sysmon_capability)
    await queue.start()
    try:
        manifest = await run_pipeline(manager, queue, cfg, sources, actor=actor,
                                      collections=collection_decls)
    finally:
        await queue.stop()
        for iid in reversed(loaded_ids):  # Reverse load order; the monitor unloads last
            try:
                manager.unload_capability(iid)
            except Exception as e:  # Best-effort teardown; never mask the run's outcome
                logger.warning(f"unload {iid} failed: {e}")

    out = (Path(args.output) if args.output
           else (ws.runs_dir if ws is not None else Path("runs")) / f"{manifest.run_id}.json")
    manifest.save(out, workspace=ws)
    done = sum(len(s.segments) for s in manifest.sources)
    print(f"run manifest: {out}")
    print(f"sources completed: {len(manifest.sources)}/{len(sources)}  segments: {done}  transcribers: {len(transcribers)}")
    if cfg.preprocessing_capability:
        print(f"preprocessing: {cfg.preprocessing_capability} ({cfg.preprocessing_task}/{cfg.preprocessing_method})")
    if cfg.diarization_capability:
        for s in manifest.sources:
            d = s.diarization or {}
            line = f"diarization [{Path(s.source_path).name}]: {d.get('status', 'skipped')}"
            if d.get("status") == "ok":
                line += f"  speakers: {d.get('speaker_count')}  turns: {d.get('turn_count')}"
            print(line)
    if cfg.graph_capability:
        for s in manifest.sources:
            print(f"graph emission [{Path(s.source_path).name}]: {s.graph}")
    return 0 if len(manifest.sources) == len(sources) else 1


def main(
    argv: Optional[List[str]] = None,  # Argument list override (None = sys.argv)
) -> int:  # Process exit code
    """CLI entry point (console script: `cjm-transcription-core`)."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s",
    )
    if args.command == "run":
        return asyncio.run(run_command(args))
    if args.command == "declare-structure":
        return asyncio.run(declare_structure_command(args))
    if args.command == "retire-collection":
        return asyncio.run(retire_collection_command(args))
    if args.command in ("add-reference", "retract-reference"):
        return asyncio.run(reference_command(args))
    if args.command == "rerun-chunk":
        return asyncio.run(rerun_chunk_command(args))
    if args.command == "add-transcript":
        return asyncio.run(add_transcript_command(args))
    if args.command == "runaway-census":
        return asyncio.run(runaway_census_command(args))
    raise SystemExit(f"unknown command: {args.command}")


def parse_transcriber_spec(
    spec: str,  # One --transcriber value: NAME[@INSTANCE_ID][:key=value,...]
) -> Dict[str, Any]:  # Load directive: {"capability", "instance_id", "config"}
    """Parse one `--transcriber` spec into a (capability, MODEL)-instance load directive.

    Grammar: `NAME[@INSTANCE_ID][:key=value[,key=value...]]`. A bare NAME keeps
    the stage-5 behavior (default instance, manifest-default config). `@INSTANCE_ID`
    names a CR-10 multi-instance load so ONE capability can host several
    (capability, MODEL) instances side by side (db200725: the whisper family /
    voxtral mini-vs-small); config overrides REQUIRE it — every non-default
    config gets its own addressable instance id. Values coerce to bool/int/float
    when they read as one, else stay strings — the manifest config_schema is the
    real validator at load time (SG-5 strict).
    """
    head, colon, cfg_part = spec.partition(":")
    name, at, instance_id = head.partition("@")
    if not name:
        raise SystemExit(f"--transcriber expects NAME[@INSTANCE_ID][:key=value,...], got {spec!r}")
    if at and not instance_id:
        raise SystemExit(f"--transcriber has a dangling '@' (empty instance id): {spec!r}")
    if not at:
        instance_id = name
    config: Dict[str, Any] = {}
    if colon:
        if not at:
            raise SystemExit(
                f"--transcriber config overrides require an explicit @INSTANCE_ID "
                f"(a non-default config needs its own addressable instance): {spec!r}")
        for pair in cfg_part.split(","):
            key, eq, value = pair.partition("=")
            if not eq or not key or not value:
                raise SystemExit(f"--transcriber config expects key=value, got {pair!r} in {spec!r}")
            if value in ("true", "false"):
                config[key] = (value == "true")
            else:
                try:
                    config[key] = int(value)
                except ValueError:
                    try:
                        config[key] = float(value)
                    except ValueError:
                        config[key] = value
    return {"capability": name, "instance_id": instance_id, "config": config}


def expand_sources(
    paths: List[str],  # CLI `audio` values: media file paths and/or directories, in order
) -> List[str]:  # Resolved media file paths (files verbatim; directories expanded recursively, sorted)
    """Expand CLI source arguments into the ordered media-file list for a run.

    Files pass through untouched (any extension — the caller asked for them by
    name); a DIRECTORY expands to every media file under it, recursively, in
    sorted-path order so folder runs stay deterministic (TUI-v0 headless slice
    be4627c7: a feedstock folder lands as one CLI arg instead of a hand-typed
    file list). Missing paths and directories with no media files refuse loudly.
    """
    out: List[str] = []
    missing: List[str] = []
    for p in paths:
        path = Path(p).resolve()
        if path.is_dir():
            found = sorted(str(f) for f in path.rglob("*")
                           if f.is_file() and f.suffix.lower() in MEDIA_SUFFIXES)
            if not found:
                raise SystemExit(f"no media files under directory: {path}")
            out.extend(found)
        elif path.exists():
            out.append(str(path))
        else:
            missing.append(str(path))
    if missing:
        raise SystemExit(f"missing audio file(s): {missing}")
    return out


# Media suffixes a DIRECTORY source expands to (explicit files pass through
# regardless — the caller asked for those by name). Shared vocabulary: the
# transcription TUI's source browser filters its listing with the same set.
MEDIA_SUFFIXES = {
    ".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav", ".wma",
    ".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm",
}


def expand_sources_with_collections(
    paths: List[str],                      # CLI `audio` values: media files and/or directories, in order
    explicit_title: Optional[str] = None,  # --collection TITLE: one human-named collection over ALL sources
    no_collection: bool = False,           # --no-collection: suppress the folder->collection proposals
    actor: str = "cli:transcribe",         # Attribution for the declarations
) -> Tuple[List[str], List[CollectionDecl]]:  # (resolved media files, collection declarations)
    """Expand CLI sources AND keep the folder-source gesture as collection
    declarations (ae3464fc) instead of throwing it away at hand-off.

    `expand_sources` stays the pure file expander; this sibling maps the
    GESTURE: each DIRECTORY arg proposes one collection — title = the folder
    name (underscores prettified for display; identity normalizes anyway),
    members = its sorted expansion, which is a real order (ordered=True). The
    proposals land status="proposed": a folder is EVIDENCE of a grouping, not
    proof. `--collection TITLE` replaces the per-folder proposals with ONE
    human-named declaration over all of the run's sources — a human naming it
    is a confirmation act (status="confirmed"; ordered only when the members
    came from a single folder expansion — never fabricate sequence from a
    hand-typed file list). `--no-collection` expands only."""
    if explicit_title and no_collection:
        raise SystemExit("--collection and --no-collection are mutually exclusive")
    files: List[str] = []
    decls: List[CollectionDecl] = []
    for p in paths:
        expanded = expand_sources([p])
        files.extend(expanded)
        if Path(p).resolve().is_dir() and not no_collection and not explicit_title:
            title = " ".join(Path(p).resolve().name.replace("_", " ").split())
            decls.append(CollectionDecl(title=title, member_paths=list(expanded),
                                        status="proposed", actor=actor, ordered=True))
    if explicit_title:
        ordered = len(paths) == 1 and Path(paths[0]).resolve().is_dir()
        decls = [CollectionDecl(title=explicit_title, member_paths=list(files),
                                status="confirmed", actor=actor, ordered=ordered)]
    return files, decls


async def declare_structure_command(
    args: argparse.Namespace,  # Parsed CLI arguments for the `declare-structure` subcommand
) -> int:  # Process exit code
    """Execute `declare-structure`: read a structure-map document and land it
    as `work_structure` property merges on the named Sources (the headless
    HITL seam for a source's part/chapter map — the human confirms the
    document, the verb journals the act; DEC 8d9de793 clause 7).

    Graph plumbing mirrors `run`: workspace resolved first (CJM_WORKSPACE
    exported), the graph capability loaded alone, --graph-db-path a
    caller-wins config. The sidecar journal is DERIVED from the effective db
    path (`sidecar_journal_path`), never configured. The graph-stack open is
    the third carried copy of the 2ce81638 shape (correction-core spine,
    hub spine) — it moves with the c3c21f99 home decision."""
    ws = resolve_workspace(explicit=getattr(args, "workspace", None))
    if ws is not None:
        os.environ["CJM_WORKSPACE"] = str(ws.root)
    doc = json.loads(Path(args.map).read_text())
    entries = structure_entries_from_map(doc)
    collection_id = args.collection_id or doc.get("collection_id")
    actor = args.actor or f"human:{getpass.getuser()}"
    if args.dry_run:
        for e in entries:
            print(f"{e['source_id'][:8]}  {e['structure']}  evidence={e['evidence']}")
        print(f"dry run: {len(entries)} sources, collection {collection_id}, actor {actor}")
        return 0
    manager = CapabilityManager(search_paths=[Path(args.manifests_dir)])
    configs = ({args.graph_capability: {"db_path": args.graph_db_path}}
               if args.graph_db_path else None)
    load_capabilities(manager, [args.graph_capability], configs=configs)
    effective = args.graph_db_path or (
        (manager.instances[args.graph_capability].config or {}).get("db_path"))
    if not effective:
        raise SystemExit("no graph db path: pass --graph-db-path, or persist one on the "
                         f"{args.graph_capability} instance in the active workspace's config store")
    journal_path = sidecar_journal_path(str(effective))
    queue = JobQueue(deps=manager)
    await queue.start()
    try:
        op = await declare_structure(queue, args.graph_capability, entries, actor,
                                     journal_path=journal_path, collection_id=collection_id)
    finally:
        await queue.stop()
        manager.unload_capability(args.graph_capability)
    print(f"declared work_structure on {op['args']['sources']} sources "
          f"(collection {collection_id}; actor {actor})")
    print(f"journal: {journal_path}")
    return 0


async def reference_command(
    args: argparse.Namespace,  # Parsed CLI arguments for `add-reference` / `retract-reference`
) -> int:  # Process exit code
    """Execute `add-reference` / `retract-reference`: attach or retract a human-added
    resource link on a Source (ruling a7ca900d (3), item ae103970) — the headless HITL
    seam for the links nothing in the audio names. Graph plumbing = the declare-structure
    shape (workspace resolved first, the graph capability loaded alone, --graph-db-path a
    caller-wins config, the sidecar journal DERIVED from the effective db path) — the
    FOURTH carried copy of the 2ce81638 open; it moves with the c3c21f99 home decision.
    `--source-id` takes the FULL Source node id (prefix resolution is a residue)."""
    ws = resolve_workspace(explicit=getattr(args, "workspace", None))
    if ws is not None:
        os.environ["CJM_WORKSPACE"] = str(ws.root)
    actor = args.actor or f"human:{getpass.getuser()}"
    manager = CapabilityManager(search_paths=[Path(args.manifests_dir)])
    configs = ({args.graph_capability: {"db_path": args.graph_db_path}}
               if args.graph_db_path else None)
    load_capabilities(manager, [args.graph_capability], configs=configs)
    effective = args.graph_db_path or (
        (manager.instances[args.graph_capability].config or {}).get("db_path"))
    if not effective:
        raise SystemExit("no graph db path: pass --graph-db-path, or persist one on the "
                         f"{args.graph_capability} instance in the active workspace's config store")
    journal_path = sidecar_journal_path(str(effective))
    queue = JobQueue(deps=manager)
    await queue.start()
    try:
        if args.command == "add-reference":
            op = await add_reference(queue, args.graph_capability, args.source_id, label=args.label,
                                     url=args.url, notes_slug=args.notes_slug, role=args.role,
                                     actor=actor, journal_path=journal_path)
            print(f"added reference {op['reference_id']} on source {args.source_id} "
                  f"(role {args.role}; actor {actor})")
        else:
            await retract_reference(queue, args.graph_capability, args.reference_id,
                                    actor=actor, journal_path=journal_path)
            print(f"retracted reference {args.reference_id} (actor {actor})")
    finally:
        await queue.stop()
        manager.unload_capability(args.graph_capability)
    print(f"journal: {journal_path}")
    return 0


async def retire_collection_command(
    args: argparse.Namespace,  # Parsed CLI arguments for `retire-collection`
) -> int:  # Process exit code
    """Execute `retire-collection` (ruling a7617bd4, item eaefebd2): resolve the
    collection by id prefix or exact title, then journal the retirement fact (or its
    reversal). Same graph plumbing as the reference verbs; nothing cascades."""
    from cjm_transcription_core.curation import list_collections, retire_collection
    ws = resolve_workspace(explicit=getattr(args, "workspace", None))
    if ws is not None:
        os.environ["CJM_WORKSPACE"] = str(ws.root)
    actor = args.actor or f"human:{getpass.getuser()}"
    manager = CapabilityManager(search_paths=[Path(args.manifests_dir)])
    configs = ({args.graph_capability: {"db_path": args.graph_db_path}}
               if args.graph_db_path else None)
    load_capabilities(manager, [args.graph_capability], configs=configs)
    effective = args.graph_db_path or (
        (manager.instances[args.graph_capability].config or {}).get("db_path"))
    if not effective:
        raise SystemExit("no graph db path: pass --graph-db-path, or persist one on the "
                         f"{args.graph_capability} instance in the active workspace's config store")
    journal_path = sidecar_journal_path(str(effective))
    queue = JobQueue(deps=manager)
    await queue.start()
    try:
        cols = await list_collections(queue, args.graph_capability)
        sel = args.collection.strip()
        hits = [c for c in cols if c["id"] == sel or c["id"].startswith(sel)] or \
               [c for c in cols if c["title"].lower() == sel.lower()]
        if len(hits) != 1:
            print(f"REFUSED: {args.collection!r} matches {len(hits)} collection(s): "
                  f"{[(c['id'][:8], c['title'], c['status']) for c in cols]}")
            return 2
        coll = hits[0]
        if not args.unretire and coll["status"] == "retired":
            print(f"REFUSED: {coll['title']!r} is already retired")
            return 2
        if args.unretire and coll["status"] != "retired":
            print(f"REFUSED: {coll['title']!r} is not retired (status {coll['status']})")
            return 2
        await retire_collection(queue, args.graph_capability, coll["id"], actor=actor,
                                reason=args.reason, journal_path=journal_path, unretire=args.unretire)
        print(f"{'restored' if args.unretire else 'retired'} collection {coll['id'][:8]} {coll['title']!r}"
              + (f" ({args.reason})" if args.reason and not args.unretire else "") + f" (actor {actor})")
    finally:
        await queue.stop()
        manager.unload_capability(args.graph_capability)
    print(f"journal: {journal_path}")
    return 0


def parse_config_overrides(
    items: Optional[List[str]],  # KEY=VALUE strings (values JSON-parsed when they parse, else kept as strings)
) -> Dict[str, Any]:  # Override dict
    """Parse repeatable KEY=VALUE config overrides (`--transcriber-config`)."""
    out: Dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--transcriber-config expects KEY=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        try:
            out[k.strip()] = json.loads(v)
        except ValueError:
            out[k.strip()] = v
    return out


def _chunk_graph_target(
    args: argparse.Namespace,  # Parsed args carrying --graph-capability / --graph-db-path
    parent: Dict[str, Any],    # The loaded parent manifest (its `graph` block is the recorded target)
) -> Tuple[str, str]:  # (graph capability name, effective db path)
    """The landing's graph target: the flags win, else the parent manifest's recorded
    emission target (the derived manifest inherits it). The explicit-db-path rule
    (027bbe56) is honoured by the manifest carrying the path it emitted into."""
    recorded = parent.get("graph") or {}
    cap = args.graph_capability or recorded.get("capability") or "cjm-capability-graph-sqlite"
    db = args.graph_db_path or recorded.get("db_path")
    if not db:
        raise SystemExit("no graph db path: the parent manifest recorded no emission target — pass --graph-db-path")
    return str(cap), str(db)


def _print_targets(
    targets: List[Tuple[int, Dict[str, Any], Dict[str, Any]]],  # (source_index, source_entry, segment_entry) rows
) -> None:
    """Print the resolved chunk targets (one line each)."""
    for si, src, seg in targets:
        print(f"  src {si:3d} seg {int(seg.get('index', -1)):4d}  {float(seg.get('start', 0)):8.1f}-{float(seg.get('end', 0)):8.1f}s  "
              f"{Path(str(src.get('source_path') or '')).name}")


async def rerun_chunk_command(
    args: argparse.Namespace,  # Parsed CLI arguments for `rerun-chunk`
) -> int:  # Process exit code (0 = every target landed)
    """Execute `rerun-chunk` (cf0b91d6 part 1; ruling 8a9b9639 chunk-targeted, never wholesale).

    Reloads ONE transcriber under the manifest's recorded instance config (+ overrides),
    runs it on each target chunk's model-input rendition with the cache bypassed, lands
    the new Transcript variant under the existing AudioSegment (SUPERSEDES the prior
    variant), journals RUN_STARTED / RUN_FINISHED under a fresh run id, and writes a
    DERIVED manifest (parent untouched; per-entry config_hash) the decomp consumes."""
    ws = resolve_workspace(explicit=getattr(args, "workspace", None))
    if ws is not None:
        os.environ["CJM_WORKSPACE"] = str(ws.root)
    actor = args.actor or f"cli:{getpass.getuser()}"
    reason = args.reason or "runaway"
    manifest_path = Path(args.manifest)
    parent = load_run_manifest(manifest_path)
    transcriber = args.transcriber
    cap = (parent.get("capabilities") or {}).get(transcriber)
    if cap is None:
        raise SystemExit(f"transcriber {transcriber!r} is not in the manifest's capabilities "
                         f"({sorted(parent.get('capabilities') or {})})")
    config = {**dict(cap.get("config") or {}), **parse_config_overrides(args.transcriber_config)}
    directive = {"capability": str(cap.get("name") or transcriber), "instance_id": transcriber, "config": config}
    graph_cap, db_path = _chunk_graph_target(args, parent)
    targets = select_chunks(parent, source=args.source, segments=args.segment)
    if not args.flagged:
        if args.dry_run:
            print(f"targets ({len(targets)}) for {transcriber} from {manifest_path}:")
            _print_targets(targets)
            return 0
    manager = CapabilityManager(search_paths=[Path(args.manifests_dir)],
                                sysmon_capability_name=args.sysmon_capability)
    load_order = ([args.sysmon_capability] if args.sysmon_capability else []) + [graph_cap]
    if not args.dry_run:
        load_order.append(directive)
    load_capabilities(manager, load_order, configs={graph_cap: {"db_path": db_path}})
    loaded_ids = [i["instance_id"] if isinstance(i, dict) else i for i in load_order]
    journal_path = sidecar_journal_path(db_path)
    queue = JobQueue(deps=manager, sysmon_capability_name=args.sysmon_capability)
    await queue.start()
    failures: List[Dict[str, Any]] = []
    landed = 0
    degenerate = 0
    run_id = new_run_id()
    out = (Path(args.output) if args.output else manifest_path.parent / f"{run_id}.json")
    try:
        if args.flagged:
            rows = await fetch_transcript_rows(queue, graph_cap)
            flagged = census_rows(rows, max_chars=args.max_chars,
                                  max_words_per_second=args.max_words_per_second,
                                  disagreement_ratio=args.disagreement_ratio, transcriber=transcriber)
            # A variant the 0.0.49 guard already truncated (`degenerate` as its ONLY
            # reason) is an ESCALATION candidate, not a re-run target — re-running it
            # under the same guard would loop again; the census keeps listing it.
            rerunnable = [f for f in flagged if set(f.get("reasons") or []) - {"degenerate"}]
            census_targets = chunks_from_census(parent, rerunnable, transcriber)
            keep = {(si, int(seg.get("index", -1))) for si, _, seg in targets}
            targets = [t for t in census_targets if (t[0], int(t[2].get("index", -1))) in keep]
            print(f"census: {len(flagged)} flagged {transcriber} variants graph-wide; "
                  f"{len(targets)} within this manifest's selection")
        if args.dry_run:
            print(f"targets ({len(targets)}) for {transcriber} from {manifest_path}:")
            _print_targets(targets)
            return 0
        if not targets:
            print("no targets — nothing to re-run")
            return 0
        queue.set_run_context(run_id=run_id, actor=actor)
        _journal_run_event(manager, SubstrateEventType.RUN_STARTED.value, run_id, actor, {
            "core": "cjm-transcription-core", "kind": "rerun-chunk",
            "parent_run_id": parent.get("run_id"), "transcriber": transcriber,
            "chunks": len(targets), "reason": reason, "graph_capability": graph_cap,
        })
        info = collect_capability_info(manager, [transcriber])
        new_hash = str((info.get(transcriber) or {}).get("config_hash") or "")
        if not new_hash:
            raise SystemExit(f"could not read the effective config hash of {transcriber!r}")
        derived = derive_manifest(parent, run_id=run_id, parent_path=manifest_path, kind="rerun-chunk")
        print(f"re-running {len(targets)} chunk(s) with {transcriber} (config {new_hash[:19]}…) -> {out}")
        for n, (si, src, seg) in enumerate(targets, 1):
            idx = int(seg.get("index", -1))
            job_id = f"{run_id}_src{si}_seg{idx:04d}_rerun"
            label = f"[{n}/{len(targets)}] src {si} seg {idx} {Path(str(src.get('source_path') or '')).name}"
            try:
                result = await submit_and_wait(
                    queue, transcriber, audio=str(seg.get("model_input_path") or ""), job_id=job_id,
                    source_start_time=float(seg.get("start", 0.0)), source_end_time=float(seg.get("end", 0.0)),
                    task="transcription", method="transcribe", control={"force": True})
                text = str(getattr(result, "text", "") or "")
                metadata = dict(getattr(result, "metadata", None) or {})
                prior = prior_config_hash(parent, seg, transcriber)
                landing = {"producer": PRODUCER_RERUN, "reason": reason, "parent_run_id": parent.get("run_id"),
                           "actor": actor, "landed_at": time.time()}
                rec = await land_chunk_transcript(
                    queue, graph_cap, src, seg, transcriber=transcriber, config_hash=new_hash, text=text,
                    metadata={**metadata, "landing": landing}, prior_config_hash=prior,
                    producer=PRODUCER_RERUN, reason=reason, actor=actor, run_id=run_id,
                    journal_path=journal_path)
                apply_chunk_update(derived, si, idx, transcriber, {
                    "job_id": job_id, "text": text, "metadata": {**metadata, "landing": landing},
                    "config_hash": new_hash,
                    "landing": {**landing, "transcript_id": rec["transcript"], "supersedes": rec["supersedes"]}})
                landed += 1
                save_manifest(derived, out, workspace=ws)  # crash-safe: every landing is on disk
                tail = metadata.get("degenerate_tail")
                if tail:
                    degenerate += 1
                words = len(text.split())
                print(f"{label}: {words} words" + (f"  DEGENERATE tail cut ({tail.get('phrase')!r} × {tail.get('repeats')})" if tail else "")
                      + f"  {'supersedes ' + str(rec['supersedes'])[:8] if rec['supersedes'] else 'no prior variant'}")
            except Exception as e:  # record + continue unless --fail-fast; the derived manifest keeps what landed
                failures.append({"source_index": si, "segment_index": idx, "error": str(e)})
                print(f"{label}: FAILED — {e}")
                if args.fail_fast:
                    break
        derived.setdefault("derivation", {})["failures"] = failures
        save_manifest(derived, out, workspace=ws)
        _journal_run_event(manager, SubstrateEventType.RUN_FINISHED.value, run_id, actor, {
            "core": "cjm-transcription-core", "kind": "rerun-chunk", "parent_run_id": parent.get("run_id"),
            "transcriber": transcriber, "chunks": len(targets), "landed": landed,
            "degenerate": degenerate, "failed": len(failures), "manifest": str(out),
        })
    finally:
        await queue.stop()
        for iid in reversed(loaded_ids):
            try:
                manager.unload_capability(iid)
            except Exception as e:  # Best-effort teardown; never mask the run's outcome
                logger.warning(f"unload {iid} failed: {e}")
    print(f"derived manifest: {out}")
    print(f"landed {landed}/{len(targets)} chunk(s); degenerate tails cut: {degenerate}; failed: {len(failures)}")
    print(f"journal: {journal_path}")
    return 0 if not failures else 1


async def add_transcript_command(
    args: argparse.Namespace,  # Parsed CLI arguments for `add-transcript`
) -> int:  # Process exit code
    """Execute `add-transcript` (cf0b91d6 part 2; ruling 9ffce5f7 (1)): land an
    operator-pasted external transcript for ONE chunk as a third transcriber
    (`<model id>/manual`, config hash = (model id, prompt hash)), the operator as actor,
    through the SAME landing a re-run uses; a derived manifest registers the transcriber
    so decomp folds it where it has text."""
    ws = resolve_workspace(explicit=getattr(args, "workspace", None))
    if ws is not None:
        os.environ["CJM_WORKSPACE"] = str(ws.root)
    actor = args.actor or f"human:{getpass.getuser()}"
    reason = args.reason or "escalation"
    manifest_path = Path(args.manifest)
    parent = load_run_manifest(manifest_path)
    targets = select_chunks(parent, source=args.source, segments=args.segment)
    if len(targets) != 1:
        raise SystemExit(f"add-transcript lands ONE chunk; the selection resolves to {len(targets)} "
                         "(give --source and one --segment)")
    (si, src, seg) = targets[0]
    idx = int(seg.get("index", -1))
    if args.text_file == "-":
        import sys
        text = sys.stdin.read()
    else:
        text = Path(args.text_file).read_text()
    text = text.strip()
    if not text:
        raise SystemExit("add-transcript: the text is empty")
    if args.prompt_file and args.prompt_hash:
        raise SystemExit("give --prompt-file OR --prompt-hash, not both")
    prompt_hash = args.prompt_hash or (prompt_hash_of(Path(args.prompt_file).read_text()) if args.prompt_file else "")
    tname = external_transcriber_name(args.model_id)
    chash = external_config_hash(args.model_id, prompt_hash)
    graph_cap, db_path = _chunk_graph_target(args, parent)
    if args.dry_run:
        print(f"would land {len(text.split())} words as {tname} (config {chash[:19]}…) on:")
        _print_targets(targets)
        return 0
    manager = CapabilityManager(search_paths=[Path(args.manifests_dir)])
    load_capabilities(manager, [graph_cap], configs={graph_cap: {"db_path": db_path}})
    journal_path = sidecar_journal_path(db_path)
    queue = JobQueue(deps=manager)
    await queue.start()
    run_id = new_run_id()
    out = (Path(args.output) if args.output else manifest_path.parent / f"{run_id}.json")
    try:
        queue.set_run_context(run_id=run_id, actor=actor)
        _journal_run_event(manager, SubstrateEventType.RUN_STARTED.value, run_id, actor, {
            "core": "cjm-transcription-core", "kind": "add-transcript", "parent_run_id": parent.get("run_id"),
            "transcriber": tname, "chunks": 1, "reason": reason, "graph_capability": graph_cap,
        })
        landing = {"producer": PRODUCER_EXTERNAL, "reason": reason, "parent_run_id": parent.get("run_id"),
                   "actor": actor, "landed_at": time.time(), "model_id": args.model_id,
                   "prompt_hash": prompt_hash, "text_source": args.text_source}
        metadata = {"model": args.model_id, "source_start_time": float(seg.get("start", 0.0)),
                    "source_end_time": float(seg.get("end", 0.0)), "landing": landing}
        prior = prior_config_hash(parent, seg, tname)
        rec = await land_chunk_transcript(
            queue, graph_cap, src, seg, transcriber=tname, config_hash=chash, text=text, metadata=metadata,
            prior_config_hash=prior, producer=PRODUCER_EXTERNAL, reason=reason, actor=actor, run_id=run_id,
            journal_path=journal_path, node_actor=actor, method="external-landing")
        derived = derive_manifest(parent, run_id=run_id, parent_path=manifest_path, kind="add-transcript")
        apply_chunk_update(derived, si, idx, tname, {
            "job_id": f"{run_id}_src{si}_seg{idx:04d}_external", "text": text, "metadata": metadata,
            "config_hash": chash, "landing": {**landing, "transcript_id": rec["transcript"], "supersedes": rec["supersedes"]}},
            capability_info={"name": tname, "version": "manual", "db_path": None, "config_hash": chash,
                             "config": {"model_id": args.model_id, "prompt_hash": prompt_hash,
                                        "text_source": args.text_source}})
        save_manifest(derived, out, workspace=ws)
        _journal_run_event(manager, SubstrateEventType.RUN_FINISHED.value, run_id, actor, {
            "core": "cjm-transcription-core", "kind": "add-transcript", "parent_run_id": parent.get("run_id"),
            "transcriber": tname, "chunks": 1, "landed": 1, "manifest": str(out),
        })
    finally:
        await queue.stop()
        manager.unload_capability(graph_cap)
    print(f"landed {len(text.split())} words as {tname} on src {si} seg {idx} "
          f"({Path(str(src.get('source_path') or '')).name}): transcript {rec['transcript']}"
          + (f" supersedes {rec['supersedes']}" if rec["supersedes"] else "")
          + f"  (nodes +{rec['nodes_added']} verified {rec['nodes_verified']}, edges +{rec['edges_added']})")
    print(f"derived manifest: {out}")
    print(f"journal: {journal_path}")
    return 0


async def runaway_census_command(
    args: argparse.Namespace,  # Parsed CLI arguments for `runaway-census`
) -> int:  # Process exit code (0 always — a report)
    """Execute `runaway-census` (cf0b91d6 part 3): the LIVE chunk variants in need of a
    better transcription, per collection — the target list of `rerun-chunk --flagged`,
    the app's flagged-chunk lane rows, and the closing evidence for check 56a802b3 (a
    zero on every live collection)."""
    ws = resolve_workspace(explicit=getattr(args, "workspace", None))
    if ws is not None:
        os.environ["CJM_WORKSPACE"] = str(ws.root)
    manager = CapabilityManager(search_paths=[Path(args.manifests_dir)])
    configs = ({args.graph_capability: {"db_path": args.graph_db_path}} if args.graph_db_path else None)
    load_capabilities(manager, [args.graph_capability], configs=configs)
    effective = args.graph_db_path or (
        (manager.instances[args.graph_capability].config or {}).get("db_path"))
    queue = JobQueue(deps=manager)
    await queue.start()
    try:
        rows = await fetch_transcript_rows(queue, args.graph_capability)
    finally:
        await queue.stop()
        manager.unload_capability(args.graph_capability)
    flagged = census_rows(rows, max_chars=args.max_chars, max_words_per_second=args.max_words_per_second,
                          disagreement_ratio=args.disagreement_ratio, transcriber=args.transcriber,
                          collections=args.collection, include_superseded=args.include_superseded,
                          include_escalated=args.include_escalated, include_retired=args.include_retired)
    scoped = [r for r in rows if (not args.collection or (r.get("collection") or "") in set(args.collection))
              and (args.include_retired or str(r.get("collection_status") or "") != "retired")]
    summary = summarize_census(flagged, scoped)
    print(f"graph: {effective}")
    print(f"transcripts: {len(rows)} total, {sum(1 for r in rows if not r.get('superseded'))} live; "
          f"flagged: {len(flagged)}" + (f" ({args.transcriber})" if args.transcriber else ""))
    for coll, c in sorted(summary.items(), key=lambda kv: (-kv[1]['flagged'], kv[0])):
        reasons = ", ".join(f"{k} {v}" for k, v in sorted(c["by_reason"].items()))
        print(f"  {coll or '(no collection)':32s} flagged {c['flagged']:4d} / live {c['live']:5d}" + (f"   {reasons}" if reasons else ""))
    for f in flagged[: args.limit]:
        print(f"  {f.get('collection') or '-':20.20s} seg {int(f.get('seg_index') or 0):4d} "
              f"{float(f.get('start') or 0):8.1f}-{float(f.get('end') or 0):8.1f}s  {str(f.get('transcriber')):26.26s} "
              f"{int(f.get('chars') or 0):7d}ch {f.get('words_per_second'):6.1f}w/s  {','.join(f.get('reasons') or [])}"
              + ("  SUPERSEDED" if f.get("superseded") else "") + ("  ESCALATED" if f.get("escalated") else "")
              + f"  {Path(str(f.get('source_path') or '')).name}")
    if len(flagged) > args.limit:
        print(f"  … {len(flagged) - args.limit} more (see --json)")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps({"graph_db_path": effective, "thresholds": {
            "max_chars": args.max_chars, "max_words_per_second": args.max_words_per_second,
            "disagreement_ratio": args.disagreement_ratio}, "summary": summary, "flagged": flagged}, indent=2))
        print(f"json: {args.json}")
    return 0


if __name__ == "__main__":  # `python -m cjm_transcription_core.cli …` = the console script (the qt lane runs the verbs this way)
    raise SystemExit(main())
