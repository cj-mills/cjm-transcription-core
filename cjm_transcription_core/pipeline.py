"""The headless transcription pipeline: VAD analysis -> boundary computation -> segment cutting -> per-segment model-input conversion -> transcription, composed over capability workers via the substrate's JobQueue. Between-stage outputs are threaded manually (run job -> read result -> submit next); the per-segment fan-out rides a CR-16 ports Composition with OutputRef bindings (this module was the real-world consumer of the original submit_sequence piping gap — pass-2 evidence in claude-docs/pass-2-evidence.md). HITL approval seams use the cheapest viable form (log + optional CLI prompt) per the cores-cluster guard-rails; each seam carries its 5-field HITL-assist annotation in its docstring."""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from cjm_capability_primitives.media_processing import (MediaArtifactResult, MediaMetadata,
                                                        MediaSegmentationResult)
from cjm_capability_primitives.source_separation import SourceSeparationResult
from cjm_capability_primitives.transcription import TranscriptionResult
from cjm_capability_primitives.vad import VADResult
from cjm_substrate.core.empirical_store import compute_config_hash
from cjm_substrate.core.journal_store import JournalEvent, SubstrateEventType
from cjm_substrate.core.manager import CapabilityManager
from cjm_substrate.core.ports import (Composition, CompositionNode, CompositionRun, NodeState,
                                      OutputRef)
from cjm_substrate.core.queue import JobQueue, JobStatus
from cjm_substrate.utils.hashing import hash_file
from cjm_transcription_core.boundaries import compute_segment_boundaries
from cjm_transcription_core.emission import emit_source_graph
from cjm_transcription_core.models import (new_run_id, PipelineConfig, RunManifest, SegmentRecord,
                                           SourceResult)

# Typed wire-kind registration (stage 2): importing the DTO classes is what
# lets the proxy's wire_decode hand this host process TYPED results. Stage 8:
# source_separation.result + media_processing.{artifact,segmentation,metadata}
# are registered too, so ffmpeg's typed convert/segment_audio/get_info results
# (and the preprocessing node's output) decode host-side for attribute access.
# The tuple keeps these SIDE-EFFECT imports referenced so the canonical emit
# cannot prune them (they are registration bindings, not name-use).

_REGISTERED_WIRE_KINDS = (VADResult, TranscriptionResult, SourceSeparationResult,
                          MediaArtifactResult, MediaSegmentationResult, MediaMetadata)

logger = logging.getLogger(__name__)


async def submit_and_wait(
    queue: JobQueue,   # Started job queue
    instance_id: str,  # Capability instance to invoke
    *,
    timeout: Optional[float] = None,  # Seconds to wait; None = no limit
    **kwargs,          # Forwarded to the capability's execute()
) -> Any:  # The completed job's result payload
    """Submit one capability job, wait for it, and return its result (raise on failure)."""
    job_id = await queue.submit(instance_id, **kwargs)
    job = await queue.wait_for_job(job_id, timeout=timeout)
    if job.status != JobStatus.completed:
        raise RuntimeError(f"{instance_id} job {job_id} {job.status}: {job.error}")
    return job.result


def normalize_vad_result(
    result: VADResult,  # Typed VAD result (wire-decoded at the proxy)
) -> Tuple[List[Dict[str, float]], float]:  # (sorted speech chunks [{start, end}], reported duration)
    """Normalize a typed VAD result into sorted speech chunks + the reported duration.

    Stage 8 (Option C): the result arrives as a `VADResult`
    with typed `TimeRange` ranges — the dict-or-object tolerance (`field_of`,
    evidence E5) and the start/start_time key-variance handling retired with
    the untyped wire. Duration comes from the result metadata; returns 0.0
    when the capability did not report one (callers fall back to an ffmpeg
    probe).
    """
    chunks = [{"start": float(r.start), "end": float(r.end)} for r in result.ranges]
    chunks.sort(key=lambda c: c["start"])
    duration = float((result.metadata or {}).get("duration", 0.0) or 0.0)
    return chunks, duration


async def convert_for_vad(
    queue: JobQueue,
    ffmpeg_id: str,            # ffmpeg capability instance id
    audio_path: str,          # Source audio to convert
    sample_rate: int = 16000, # Target rate (Silero supports 8k/16k)
    channels: int = 1,        # Mono
    force: bool = False,      # Per-call cache-bypass (rides CallEnvelope.control)
) -> str:  # Path to the model-ready (mono, target-rate, soxr-resampled) audio
    """Convert a source to MODEL-READY audio for VAD via the ffmpeg `convert` action.

    Stage 8 (Option C): VAD operates on model-ready audio — the in-tool librosa
    decode/resample retired, so the whole source is converted once here (ffmpeg,
    soxr resampler) and the returned path feeds analyze_vad. Boundaries come back
    in seconds and apply to the ORIGINAL source for cutting."""
    result = await submit_and_wait(
        queue, ffmpeg_id,
        task="media_processing", method="convert", input_path=audio_path,
        output_format="wav", sample_rate=sample_rate, channels=channels,
        control={"force": force},
    )
    # Typed MediaArtifactResult (stage-8 wire layer): attribute access.
    return str(getattr(result, "output_path", "") or "")


async def analyze_vad(
    queue: JobQueue,
    vad_id: str,          # VAD capability instance id
    audio_path: str,      # MODEL-READY audio file to analyze (converted upstream)
    force: bool = False,  # Bypass the VAD capability's cache (rides CallEnvelope.control)
) -> Tuple[List[Dict[str, float]], float]:  # (speech chunks, reported duration)
    """Run VAD analysis on one model-ready audio file (task channel: vad/detect_speech)."""
    result = await submit_and_wait(queue, vad_id, audio=audio_path,
                                   task="vad", method="detect_speech",
                                   control={"force": force})
    return normalize_vad_result(result)


async def probe_duration(
    queue: JobQueue,
    ffmpeg_id: str,   # ffmpeg capability instance id
    audio_path: str,  # Audio file to probe
) -> float:  # Duration in seconds (0.0 when the probe fails to report one)
    """Probe a media file's duration via the ffmpeg capability's `get_info` action."""
    info = await submit_and_wait(queue, ffmpeg_id,
                                 task="media_processing", method="get_info", file_path=audio_path)
    # get_info returns a typed MediaMetadata (the uncached media_processing probe).
    return float(getattr(info, "duration", 0.0) or 0.0)


async def cut_segments(
    queue: JobQueue,
    ffmpeg_id: str,                      # ffmpeg capability instance id
    audio_path: str,                     # Source audio to cut
    boundaries: List[Dict[str, float]],  # [{start, end}, ...] from compute_segment_boundaries
    force: bool = False,                 # Per-call cache-bypass (rides CallEnvelope.control)
) -> Tuple[List[Any], str]:  # (typed MediaSegments from ffmpeg, batch_key)
    """Cut the source audio at the computed boundaries via ffmpeg `segment_audio`."""
    result = await submit_and_wait(
        queue, ffmpeg_id,
        task="media_processing", method="segment_audio", input_path=audio_path,
        boundaries=boundaries, control={"force": force},
    )
    # Typed MediaSegmentationResult: .segments holds typed MediaSegments.
    segments = list(getattr(result, "segments", None) or [])
    batch_key = str(getattr(result, "batch_key", "") or "")
    if not segments:
        raise RuntimeError(f"segment_audio produced no segments for {audio_path}: {result!r}")
    return segments, batch_key


def build_segment_composition(
    raw_segments: List[Any],  # Typed MediaSegments from ffmpeg segment_audio
    run_id: str,           # Run id (prefixes per-segment provenance job ids)
    source_index: int,     # Position of this source within the run
    ffmpeg_id: str,        # ffmpeg capability instance id
    transcriber_ids: List[str],  # Transcription capability instance ids (one or more)
    sample_rate: int = 16000,  # Model-input sample rate
    channels: int = 1,         # Model-input channel count
    force: bool = False,       # Per-call cache-bypass control flag
    preprocessing_capability: Optional[str] = None,    # Opt-in audio-preprocessing capability (None = off)
    preprocessing_task: str = "source_separation", # Task-channel task for the preprocessing step
    preprocessing_method: str = "separate_vocals", # Task-channel method for the preprocessing step
) -> Tuple[Composition, List[Dict[str, Any]]]:  # (composition, per-segment meta rows)
    """Build the per-source fan-out composition: N independent [preprocess→]convert→(T× transcribe) pipes.

    Host-constructed fan-out (stage-3 ratified shape): the host computes every
    per-item kwarg statically; the only execution-time unknowns — the hashed
    output paths of the preprocessing + convert steps — flow through `OutputRef`
    bindings. Stage 5 (dual-transcriber = the named parallel-port adopter): each
    segment's convert output fans out to ONE transcribe node PER TRANSCRIBER.

    Stage 8 (opt-in preprocessing): when `preprocessing_capability` is set, a
    preprocessing node (e.g. Demucs vocals isolation) runs FIRST on the FULL-BAND
    raw segment, and the model-input convert consumes ITS output (vocals →
    convert → transcribe). The convert output — recorded as `model_input_path` —
    is therefore the vocals-isolated model-ready WAV, which decomp's VAD+FA
    inherit for free. Routed through the task channel by (preprocessing_task,
    preprocessing_method) so the slot is FAMILY-AGNOSTIC.
    """
    nodes: List[CompositionNode] = []
    metas: List[Dict[str, Any]] = []
    for seg in raw_segments:
        # Typed MediaSegment (stage-8 media_processing.segmentation): attr access.
        idx = int(seg.index)
        seg_path = str(seg.output_path)
        start = float(seg.start)
        end = float(seg.end)

        # Stage 8: optional preprocessing on the FULL-BAND raw segment, BEFORE the
        # model-input convert. The convert then consumes the preprocessed output
        # (vocals), so model_input_path becomes the vocals-isolated WAV.
        convert_input: Any = seg_path
        separate_node: Optional[str] = None
        if preprocessing_capability:
            separate_node = f"separate_{idx:04d}"
            nodes.append(CompositionNode(separate_node, preprocessing_capability, {
                "audio": seg_path,
            }, task_name=preprocessing_task, method=preprocessing_method,
               control={"force": force}))
            convert_input = OutputRef(separate_node, "output_path")

        conv = f"convert_{idx:04d}"
        nodes.append(CompositionNode(conv, ffmpeg_id, {
            "input_path": convert_input,
            "output_format": "wav", "sample_rate": sample_rate, "channels": channels,
        }, task_name="media_processing", method="convert", control={"force": force}))
        transcribe_nodes: Dict[str, str] = {}
        job_ids: Dict[str, str] = {}
        for ti, transcriber_id in enumerate(transcriber_ids):
            tr = f"transcribe_t{ti}_{idx:04d}"
            job_id = f"{run_id}_src{source_index}_seg{idx:04d}_t{ti}"
            nodes.append(CompositionNode(tr, transcriber_id, {
                "audio": OutputRef(conv, "output_path"),
                # job_id / source_*_time ride the CR-15 identity/provenance kwarg
                # channel; force is the per-call control flag (CR-15 category 4).
                "job_id": job_id,
                "source_start_time": start,
                "source_end_time": end,
            }, task_name="transcription", method="transcribe", control={"force": force}))
            transcribe_nodes[transcriber_id] = tr
            job_ids[transcriber_id] = job_id
        metas.append({"index": idx, "segment_path": seg_path, "start": start,
                      "end": end, "job_ids": job_ids, "convert_node": conv,
                      "separate_node": separate_node,
                      "transcribe_nodes": transcribe_nodes})
    return Composition(nodes=nodes), metas


def records_from_composition(
    crun: CompositionRun,          # Terminal composition run
    metas: List[Dict[str, Any]],   # Meta rows from build_segment_composition
) -> List[SegmentRecord]:  # Ordered per-segment records
    """Fold a completed segment composition back into SegmentRecords.

    Raises on a non-completed run, surfacing the failed nodes' structured
    errors — under fail_fast a single segment failure stops the source,
    matching the pre-ports loop where the first raise aborted the source.
    Stage 5: each record carries per-transcriber `transcripts` (symmetric
    variants; authority is the decomp consumer's choice).
    """
    if crun.status != NodeState.completed:
        failed = {nid: str(nr.error) for nid, nr in crun.node_runs.items()
                  if nr.state == NodeState.failed}
        raise RuntimeError(f"segment composition {crun.status.value}: {failed}")
    results = crun.results_by_node()
    records: List[SegmentRecord] = []
    for m in metas:
        # Typed MediaArtifactResult (stage-8 wire layer): attribute access.
        wav_path = str(getattr(results[m["convert_node"]], "output_path", "") or "")
        transcripts: Dict[str, Dict[str, Any]] = {}
        for transcriber_id, node_id in m["transcribe_nodes"].items():
            tr = results[node_id]
            # Typed TranscriptionResult (stage-2 wire layer): attribute access.
            transcripts[transcriber_id] = {
                "job_id": m["job_ids"][transcriber_id],
                "text": str(tr.text or ""),
                "metadata": dict(tr.metadata or {}),
            }
        records.append(SegmentRecord(
            index=m["index"], start=m["start"], end=m["end"],
            duration=m["end"] - m["start"],
            segment_path=m["segment_path"], model_input_path=wav_path,
            transcripts=transcripts,
        ))
    return records


def tier1_segment_checks(
    boundaries: List[Dict[str, float]],  # Computed segment boundaries
    max_segment_duration: float,         # The configured wall-clock cap
    chunk_count: int,                    # VAD speech-chunk count
) -> List[str]:  # Human-readable warnings (empty = all clear)
    """Tier-1 deterministic pre-filters for the boundary-review seam (no AI)."""
    warnings: List[str] = []
    if chunk_count == 0:
        warnings.append("VAD detected NO speech chunks — source may be silent or non-speech")
    for i, b in enumerate(boundaries[:-1]):
        if (b["end"] - b["start"]) > max_segment_duration:
            warnings.append(
                f"non-final segment {i} exceeds max duration: {b['end'] - b['start']:.1f}s"
            )
    if boundaries:
        final = boundaries[-1]
        if (final["end"] - final["start"]) > 2 * max_segment_duration:
            warnings.append(
                f"final segment unusually long ({final['end'] - final['start']:.1f}s) — long trailing silence?"
            )
    return warnings


def tier1_transcript_checks(
    segments: List[SegmentRecord],  # Transcribed segments for one source
) -> List[str]:  # Human-readable warnings (empty = all clear)
    """Tier-1 deterministic pre-filters for the transcript-review seam (no AI)."""
    warnings: List[str] = []
    for s in segments:
        for tname, tr in s.transcripts.items():
            text = str(tr.get("text") or "")
            if not text.strip():
                warnings.append(
                    f"segment {s.index} [{tname}] produced EMPTY text ({s.duration:.1f}s of audio)"
                )
            elif s.duration > 30 and len(text) < 20:
                warnings.append(
                    f"segment {s.index} [{tname}]: suspiciously short text ({len(text)} chars for {s.duration:.1f}s)"
                )
    return warnings


def confirm_seam(
    seam: str,                 # Seam label, e.g. "boundary-review"
    summary_lines: List[str],  # What the operator is being asked to accept
    warnings: List[str],       # Tier-1 warnings (logged prominently)
    assume_yes: bool = False,  # Headless mode: accept without prompting
) -> bool:  # True = proceed, False = operator aborted
    """HITL approval seam in its cheapest viable form (log + optional CLI prompt).

    Per-seam capability annotation (HITL-assist methodology, 5 fields):
      1. signal: per-source summaries + Tier-1 warnings
      2. deterministic pre-filter: the tier1_* check functions (no AI)
      3. modality-bridge candidate: spectrogram render for boundary sanity (future Tier 2)
      4. authoritative verifier: re-transcribe-and-compare via a second capability (future Tier 3)
      5. flywheel capture: accept/abort decisions are logged; durable capture is
         a pass-2 seam-contract concern, not solved here

    NOTE: input() blocks the event loop — acceptable because seams sit between
    stages with no jobs in flight; the pass-2 seam contract needs an async shape.
    """
    for line in summary_lines:
        logger.info(f"[{seam}] {line}")
    for w in warnings:
        logger.warning(f"[{seam}] {w}")
    if assume_yes:
        logger.info(f"[{seam}] auto-accepted (assume_yes)")
        return True
    reply = input(f"[{seam}] proceed? [Y/n] ").strip().lower()
    accepted = reply in ("", "y", "yes")
    logger.info(f"[{seam}] {'accepted' if accepted else 'ABORTED'} by operator")
    return accepted


async def run_source(
    queue: JobQueue,
    cfg: PipelineConfig,  # Run configuration
    source_path: str,     # Source audio file
    run_id: str,          # Run id (prefixes per-segment job ids)
    source_index: int,    # Position of this source within the run
) -> Optional[SourceResult]:  # None when the operator aborts at a seam
    """Run the full pipeline for one source: VAD → boundaries → cut → [preprocess →] convert → transcribe."""
    t0 = time.time()
    logger.info(f"[src {source_index}] {source_path}")

    # 0. Content-address the source (the Source node identity input; stage 5).
    content_hash = hash_file(source_path)

    # 1. Convert the whole source to model-ready audio (mono, target rate, soxr)
    #    for VAD, then run VAD on it. Stage 8: VAD operates on model-ready audio
    #    (the in-tool librosa decode/resample retired). Boundaries come back in
    #    seconds, so they apply to the ORIGINAL source for cutting (step 4).
    #    (duration falls back to an ffmpeg probe when unreported). NOTE: the
    #    chunking VAD runs on the RAW (converted) source — preprocessing applies
    #    to the per-segment transcription/decomp path, not this chunking pass.
    vad_audio = await convert_for_vad(queue, cfg.ffmpeg_capability, source_path,
                                      sample_rate=cfg.sample_rate, channels=cfg.channels,
                                      force=cfg.force)
    chunks, duration = await analyze_vad(queue, cfg.vad_capability, vad_audio, force=cfg.force)
    if duration <= 0:
        duration = await probe_duration(queue, cfg.ffmpeg_capability, source_path)
    logger.info(f"[src {source_index}] VAD: {len(chunks)} speech chunks over {duration:.1f}s")

    # 2. Boundaries (pure logic)
    boundaries = compute_segment_boundaries(chunks, cfg.max_segment_duration, duration)

    # 3. HITL seam: boundary review
    if boundaries:
        longest = max(b["end"] - b["start"] for b in boundaries)
        summary = [f"{Path(source_path).name}: {len(boundaries)} segment(s), longest {longest:.1f}s"]
    else:
        summary = [f"{Path(source_path).name}: no segments computed"]
    if not confirm_seam(
        "boundary-review", summary,
        tier1_segment_checks(boundaries, cfg.max_segment_duration, len(chunks)),
        assume_yes=cfg.assume_yes,
    ):
        return None

    # 4. Cut the source at the boundaries
    raw_segments, batch_key = await cut_segments(queue, cfg.ffmpeg_capability, source_path, boundaries,
                                                 force=cfg.force)

    # 5. Per segment: [preprocess →] convert → (T× transcribe) as ONE composition
    # of N independent pipes (CR-16 ports; stage-5 dual-transcriber fan-out;
    # stage-8 opt-in preprocessing). When cfg.preprocessing_capability is set, each
    # segment's full-band audio is vocals-isolated before the model-input convert.
    comp, metas = build_segment_composition(
        raw_segments, run_id, source_index,
        cfg.ffmpeg_capability, cfg.transcriber_capabilities,
        sample_rate=cfg.sample_rate, channels=cfg.channels, force=cfg.force,
        preprocessing_capability=cfg.preprocessing_capability,
        preprocessing_task=cfg.preprocessing_task,
        preprocessing_method=cfg.preprocessing_method,
    )
    comp_id = await queue.submit_composition(comp)
    crun = await queue.wait_for_composition(comp_id)
    records = records_from_composition(crun, metas)

    # 5b. Content-address the model-input WAVs (AudioSegment provenance; the
    # audio of record, E14 — graph emission + downstream slice refs hang off it).
    # Under preprocessing this is the VOCALS-isolated model-input (its hash thus
    # differs from a non-preprocessed run — the separate-graph requirement).
    for r in records:
        if r.model_input_path:
            r.model_input_hash = hash_file(r.model_input_path)
        for tname, tr in r.transcripts.items():
            logger.info(f"[src {source_index}] seg {r.index} [{tname}]: {len(tr.get('text') or '')} chars")

    # 6. HITL seam: transcript review
    total_chars = {t: sum(len(r.transcripts.get(t, {}).get("text") or "") for r in records)
                   for t in cfg.transcriber_capabilities}
    chars_summary = "  ".join(f"{t}: {n} chars" for t, n in total_chars.items())
    if not confirm_seam(
        "transcript-review",
        [f"{Path(source_path).name}: {len(records)} segment(s)  {chars_summary}"],
        tier1_transcript_checks(records),
        assume_yes=cfg.assume_yes,
    ):
        return None

    logger.info(f"[src {source_index}] done in {time.time() - t0:.1f}s")
    return SourceResult(
        source_path=source_path, duration=duration,
        vad_chunk_count=len(chunks), batch_key=batch_key,
        content_hash=content_hash, segments=records,
    )


def collect_capability_info(
    manager: CapabilityManager,   # Manager holding the loaded capabilities
    instance_ids: List[str],  # Instance ids to record
) -> Dict[str, Dict[str, Any]]:  # instance_id -> {name, version, db_path, config_hash}
    """Record capability identity + data-DB pointers for the run manifest (provenance).

    Stage 5: also records each capability's EFFECTIVE config hash (the same
    `compute_config_hash` the empirical store keys on) — Transcript node
    identity is (audio segment, transcriber, config_hash), so the manifest must
    carry the hash for downstream id recomputation. `db_path` prefers the
    effective config over the manifest default (the D19 lesson). Stage 6 (0.2.1): the EFFECTIVE config
    dict is recorded READABLY beside its hash -- the I8 lesson (a persisted
    stress config was only diagnosable by hash archaeology; bundle recipients
    should read model identity directly).
    """
    info: Dict[str, Dict[str, Any]] = {}
    for iid in instance_ids:
        meta = (getattr(manager, "capabilities", {}) or {}).get(iid)
        if meta is None:
            continue
        manifest = getattr(meta, "manifest", {}) or {}
        current_config: Dict[str, Any] = {}
        try:
            proxy = manager.get_capability(iid)
            if proxy is not None:
                current_config = proxy.get_current_config() or {}
        except Exception as e:  # Best-effort: identity recording must not fail the run
            logger.warning(f"collect_capability_info: get_current_config({iid}) failed: {e}")
        info[iid] = {
            "name": meta.name,
            "version": getattr(meta, "version", None),
            "db_path": current_config.get("db_path") or manifest.get("db_path"),
            "config_hash": compute_config_hash(current_config),
            "config": current_config,
        }
    return info


def _journal_run_event(
    manager: CapabilityManager,  # Manager owning the journal store
    event_type: str,         # SubstrateEventType value (run_started / run_finished)
    run_id: str,             # This run's manifest id
    actor: Optional[str],    # Who/what initiated the run
    payload: Dict[str, Any], # Run-level structured detail
) -> None:
    """Append a host-tier run event to the journal (CR-14 follow-up).

    The cores are the trusted host writer class: RUN_STARTED/RUN_FINISHED
    bracket the run so the run manifest (same run_id) links to every job row
    the run produced. No-op when the manager has no journal store (test
    doubles); append failures stay LOUD (journal contract).
    """
    journal = getattr(manager, "journal_store", None)
    if journal is None:
        return
    journal.append(JournalEvent(
        event_type=event_type, run_id=run_id, actor=actor, payload=payload))


async def run_pipeline(
    manager: CapabilityManager,  # Manager with the capabilities loaded
    queue: JobQueue,         # Started job queue
    cfg: PipelineConfig,     # Run configuration
    sources: List[str],      # Source audio paths, in order
    run_id: Optional[str] = None,  # Override run id (default: generated)
    actor: Optional[str] = None,   # Who/what initiated (journal attribution; CLI default cli:<user>)
) -> RunManifest:  # Manifest of everything the run produced
    """Run the transcription pipeline over the given sources, in order.

    An operator abort at any seam stops the run; the manifest holds the sources
    completed so far (capability-side caches make re-runs cheap). With
    `cfg.graph_capability` set, each completed source EMITS the graph root
    (Source → AudioSegment → AudioRendition → Transcript; CR-18 revolution 2)
    idempotently — re-runs verify-collide instead of duplicating.
    """
    run_id = run_id or new_run_id()
    # CR-14 follow-up: queue-scoped run context — every job submitted in this
    # run carries run_id/actor into its journal rows + worker diagnostics
    # (run-manifest <-> journal linkage); the run itself is bracketed by
    # RUN_STARTED/RUN_FINISHED host-tier rows.
    queue.set_run_context(run_id=run_id, actor=actor)
    _journal_run_event(manager, SubstrateEventType.RUN_STARTED.value, run_id, actor, {
        "core": "cjm-transcription-core",
        "sources": [str(s) for s in sources],
        "transcribers": list(cfg.transcriber_capabilities),
        "preprocessing_capability": cfg.preprocessing_capability,
        "graph_capability": cfg.graph_capability,
    })
    capability_ids = ([cfg.vad_capability, cfg.ffmpeg_capability]
                  + ([cfg.preprocessing_capability] if cfg.preprocessing_capability else [])
                  + list(cfg.transcriber_capabilities)
                  + ([cfg.graph_capability] if cfg.graph_capability else []))
    manifest = RunManifest(
        run_id=run_id,
        created_at=time.time(),
        config=cfg.to_dict(),
        capabilities=collect_capability_info(manager, capability_ids),
    )
    if cfg.graph_capability:
        manifest.graph = {
            "capability": cfg.graph_capability,
            "db_path": (manifest.capabilities.get(cfg.graph_capability) or {}).get("db_path"),
        }
    transcriber_config_hashes = {
        t: str((manifest.capabilities.get(t) or {}).get("config_hash") or "")
        for t in cfg.transcriber_capabilities
    }
    # AudioRendition era: the preprocessing chain that produced the model-inputs
    # ([] = raw convert-only). It is the AudioRendition IDENTITY input — recorded
    # per-source in the manifest so a downstream extender recomputes the rendition
    # id (and the Transcript/Segment ids keyed on it) with no search. One step
    # today: "<task>:<capability>@<effective config hash>".
    preprocessing_chain: List[str] = []
    if cfg.preprocessing_capability:
        pp = manifest.capabilities.get(cfg.preprocessing_capability) or {}
        preprocessing_chain = [f"{cfg.preprocessing_task}:{cfg.preprocessing_capability}@{pp.get('config_hash') or ''}"]
    status = "completed"
    try:
        for i, src in enumerate(sources):
            result = await run_source(queue, cfg, str(src), run_id, i)
            if result is None:
                logger.warning(
                    f"run {run_id}: aborted at source {i} ({src}); manifest holds {i} source(s)"
                )
                status = "aborted"
                break
            result.chain = list(preprocessing_chain)  # record the rendition-identity chain in the manifest
            if cfg.graph_capability:
                result.graph = await emit_source_graph(
                    queue, cfg.graph_capability, result, transcriber_config_hashes, run_id,
                    chain=preprocessing_chain,
                )
                logger.info(f"[src {i}] graph emission: {result.graph}")
            manifest.sources.append(result)
    except BaseException as e:
        # The journal exists for exactly this row: a run that DIED records
        # how far it got (failures stop being the unattributed case).
        _journal_run_event(manager, SubstrateEventType.RUN_FINISHED.value, run_id, actor, {
            "core": "cjm-transcription-core", "status": "failed", "error": repr(e),
            "sources_completed": len(manifest.sources), "sources_total": len(sources),
        })
        raise
    _journal_run_event(manager, SubstrateEventType.RUN_FINISHED.value, run_id, actor, {
        "core": "cjm-transcription-core", "status": status,
        "sources_completed": len(manifest.sources), "sources_total": len(sources),
        "segments": sum(len(s.segments) for s in manifest.sources),
    })
    return manifest
