"""Data shapes for the transcription pipeline: run configuration + the run-manifest result containers. The run manifest is the pipeline's durable output record: which sources were processed, how they were segmented, and where each segment's transcription landed (capability data DBs remain the authoritative text store; the manifest records the run's shape + provenance pointers). It is a deliberate proto-bundle — the CR-20 provenance-bundle infrastructure is expected to absorb/replace it."""

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from cjm_substrate.core.workspace import relativize_recorded


@dataclass
class PipelineConfig:
    """Configuration for one transcription pipeline run."""
    vad_capability: str = "cjm-capability-silero-vad"               # VAD capability instance id
    ffmpeg_capability: str = "cjm-capability-ffmpeg"                # Convert/segment capability instance id
    transcriber_capabilities: List[str] = field(                       # Transcription capability instance ids (one or more; stage-5 dual-transcriber)
        default_factory=lambda: ["cjm-capability-whisper"])
    graph_capability: Optional[str] = None   # Graph-storage capability for Source/AudioSegment/Transcript emission (None = no emission)
    graph_db_path: Optional[str] = None  # Explicit graph DB path override (caller-wins config, C8/F10)
    # Opt-in audio preprocessing (stage 8 — Demucs source separation is the first
    # family). When set, each ~5-min segment is preprocessed BEFORE the model-input
    # convert: full-band segment -> separate_vocals -> convert -> transcribe / (decomp) VAD+FA.
    # The slot is FAMILY-AGNOSTIC: it routes through the task channel by
    # (preprocessing_task, preprocessing_method), so a future preprocessing family
    # (e.g. speech-enhancement) drops in by changing those, not the pipeline.
    preprocessing_capability: Optional[str] = None        # Preprocessing capability instance id (None = preprocessing OFF)
    preprocessing_task: str = "source_separation"     # Task-channel task for the preprocessing step
    preprocessing_method: str = "separate_vocals"     # Task-channel method for the preprocessing step
    # Default-ON speaker diarization (2026-07-26): anonymous full-source turns
    # acquired BESIDE transcription (analysis, the silero-vad family shape —
    # never preprocessing; a9cadfec / DEC 7a44a808). SOURCE-KEYED: the turns
    # artifact lands under <diarization_root>/diarization/ by source content
    # hash, so existing decomposed spines inherit turns with no respine.
    # Failures are contained — signal acquisition must not kill a run.
    diarization_capability: Optional[str] = "cjm-capability-pyannote"  # Diarization capability instance id (None = diarization OFF)
    diarization_task: str = "speaker_diarization"  # Task-channel task for the diarization step
    diarization_method: str = "diarize"            # Task-channel method for the diarization step
    diarization_root: Optional[str] = None         # Turns-artifact root (workspace root; None = no artifact persistence)
    max_segment_duration: float = 220.0  # Wall-clock cap per segment in seconds (pre-emptive cuts). 220 keeps each segment's forced-alignment input clear of qwen3-FA's ~240-250s degeneracy cliff (300 sat AT the cliff -> tail over-assignment; FA over-assignment investigation 2026-06-16; 220 chosen over 240 to absorb the soft-cap silence-gap overshoot)
    sample_rate: int = 16000             # Model-input sample rate for the per-segment convert step
    channels: int = 1                    # Model-input channel count
    force: bool = False                  # Bypass capability-side caches (VAD + transcription + preprocessing)
    assume_yes: bool = False             # Auto-accept HITL seams (headless / corpus-generation mode)

    def to_dict(self) -> Dict[str, Any]:  # Plain-dict snapshot for the run manifest
        """Serialize to a plain dict."""
        return asdict(self)


@dataclass
class SegmentRecord:
    """One segment of a source audio file, with per-transcriber transcripts.

    Manifest schema 0.2.0: the single text/job_id pair became `transcripts`
    keyed by transcriber capability name — transcription emits SYMMETRIC
    variants; the authority designation is the decomp consumer's choice
    (stage-5 ratified design). `model_input_hash` content-addresses the
    model-input WAV (the audio of record, E14) for graph emission identity."""
    index: int              # 0-based position within the source
    start: float            # Segment start in source-audio seconds
    end: float              # Segment end in source-audio seconds
    duration: float         # Wall-clock segment duration in seconds
    segment_path: str       # Cut audio file (source codec) from ffmpeg `segment_audio`
    model_input_path: str   # Model-ready WAV from the per-segment `convert` step
    model_input_hash: str = ""  # Content hash over the model-input WAV ("algo:hexdigest")
    transcripts: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # transcriber -> {job_id, text, metadata}

    def to_dict(self) -> Dict[str, Any]:  # Plain-dict form for the run manifest
        """Serialize to a plain dict."""
        return asdict(self)


@dataclass
class SourceResult:
    """Pipeline result for one source audio file."""
    source_path: str        # Original input audio path
    duration: float         # Source duration in seconds
    vad_chunk_count: int    # Number of speech chunks VAD detected
    batch_key: str          # ffmpeg `segment_audio` batch key linking the cut files
    content_hash: str = ""  # Content hash over the source file (Source node identity input)
    segments: List[SegmentRecord] = field(default_factory=list)  # Ordered transcribed segments
    chain: List[str] = field(default_factory=list)  # Preprocessing chain that produced the model-inputs ([] = raw convert-only); AudioRendition identity input — extenders recompute the rendition id from it
    graph: Optional[Dict[str, Any]] = None  # Emission record: {source_node_id, nodes_added, nodes_verified, edges_added} (None = not emitted)
    diarization: Optional[Dict[str, Any]] = None  # Diarization record: {capability, config_hash, status, turn_count, speaker_count, turns_path} (None = diarization OFF this run)

    def to_dict(self) -> Dict[str, Any]:  # Plain-dict form for the run manifest
        """Serialize to a plain dict with nested segments."""
        return {
            "source_path": self.source_path,
            "duration": self.duration,
            "vad_chunk_count": self.vad_chunk_count,
            "batch_key": self.batch_key,
            "content_hash": self.content_hash,
            "segments": [s.to_dict() for s in self.segments],
            "chain": list(self.chain),
            "graph": self.graph,
            "diarization": self.diarization,
        }


@dataclass
class RunManifest:
    """Durable record of one pipeline run (proto-bundle; see CR-20).

    Schema 0.5.0 adds per-source `diarization` — the speaker-diarization
    provenance record for the default-on turns-acquisition rung (the turns
    themselves persist SOURCE-KEYED under <workspace>/diarization/, not in
    the manifest). Schema 0.4.0 adds `collections` — the run's collection declarations
    (ae3464fc; additive, [] when none ride the run). Schema 0.3.0
    (AudioRendition era): per-source `chain` records the
    preprocessing chain that produced the model-inputs ([] = raw convert-only),
    so a downstream extender can RECOMPUTE the deterministic AudioRendition node
    id (and the Transcript/Segment ids keyed on it) with no search. Builds on
    0.2.0's per-segment `transcripts` keyed by transcriber + source
    `content_hash` + per-segment `model_input_hash` + capability `config_hash`."""
    run_id: str                       # Unique run identifier
    created_at: float                 # Unix timestamp at run start
    config: Dict[str, Any]            # PipelineConfig snapshot
    capabilities: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # instance_id -> {name, version, db_path, config_hash}
    sources: List[SourceResult] = field(default_factory=list)         # Per-source results, input order
    graph: Optional[Dict[str, Any]] = None  # Emission target: {capability, db_path} (None = no emission this run)
    collections: List["CollectionDecl"] = field(default_factory=list)  # Collection declarations riding this run (0.4.0; [] = none)

    FORMAT: str = field(default="cjm-transcription-core/run-manifest", repr=False)  # Manifest format tag
    VERSION: str = field(default="0.5.0", repr=False)                               # Manifest schema version

    def to_dict(self) -> Dict[str, Any]:  # Plain-dict form for JSON serialization
        """Serialize to a plain dict with nested sources."""
        return {
            "format": self.FORMAT,
            "version": self.VERSION,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "config": self.config,
            "capabilities": self.capabilities,
            "sources": [s.to_dict() for s in self.sources],
            "graph": self.graph,
            "collections": [c.to_dict() for c in self.collections],
        }

    def save(
        self,
        path: Union[str, Path],  # Destination JSON file (parent dirs created)
        workspace=None,  # Active Workspace; owned paths record as ${WS}/<rel> (5daadfc4 rung f)
    ) -> Path:  # The written path
        """Write the manifest as pretty-printed JSON.

        With `workspace`, recorded paths under its root take the ${WS}/ token
        form (relativize_recorded), so the manifest relocates with the
        workspace; readers resolve via resolve_recorded_tree at load."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(relativize_recorded(self.to_dict(), workspace), indent=2))
        return out


def new_run_id() -> str:  # e.g. "run_20260607_153000_1a2b3c4d"
    """Generate a unique, sortable run id."""
    return f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


@dataclass
class CollectionDecl:
    """A collection declaration riding a run (ae3464fc: the folder-source
    gesture IS a collection-shaped declaration — captured at hand-off instead
    of thrown away).

    `status` follows the actor criterion: a human who accepted/renamed/selected
    the collection at launch has confirmed it; untouched tool defaults (a
    headless folder run) stay "proposed". `ordered` is True only when the
    members carry a real order (a folder's sorted expansion) — never fabricate
    sequence at capture. Members are the run-source paths the declaration
    covers; emission maps them to Source node ids from the completed results,
    so an aborted run files exactly the members that finished."""
    title: str                   # Collection display title (identity input via the schema's normalization)
    member_paths: List[str] = field(default_factory=list)  # Resolved member source paths (declaration order)
    status: str = "proposed"     # "proposed" (tool inference) | "confirmed" (human act)
    actor: str = "cli:transcribe"  # Who declared it (attribution)
    ordered: bool = False        # True = members carry a proposed order (folder expansion)

    def to_dict(self) -> Dict[str, Any]:  # Plain-dict form for the run manifest
        """Serialize to a plain dict."""
        return asdict(self)
