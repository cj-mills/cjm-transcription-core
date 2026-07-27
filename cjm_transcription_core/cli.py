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
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from cjm_substrate.core.manager import CapabilityManager
from cjm_substrate.core.queue import JobQueue
from cjm_substrate.core.workspace import resolve_workspace
from cjm_transcription_core.models import CollectionDecl, PipelineConfig
from cjm_transcription_core.pipeline import run_pipeline

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
