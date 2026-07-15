# cjm-transcription-core

<!-- generated from the context graph by `cjm-context-graph readme` — do not edit by hand; edit the graph (the urge to hand-edit = move it on-graph) -->

A frontend-agnostic core for the audio transcription workflow — composes isolated capability workers (audio conversion, VAD segmentation, batch transcription, persistence) into a headless pipeline, with a CLI as its first driver.

## Modules

- **`cjm_transcription_core.boundaries`** — Wall-clock-aware segment boundary computation: group VAD speech chunks into segments cut at silence-gap midpoints. Pure logic — no capability calls. Final home of the algorithm originally validated in cjm-transcription-audio-segment's AudioSegmentService.compute_segment_boundaries (that library is retired to cj-mills_deferred/).
- **`cjm_transcription_core.cli`** — The CLI driver — the workflow core's first (and currently only) frontend.
- **`cjm_transcription_core.emission`** — Graph-root emission (CR-18 revolution 2): a completed source EMITS Source -> AudioSegment -> Transcript into the shared context graph — the graph BEGINS at transcription (where-graph-begins resolution: ingestion is the first EXTENDER that plants the root). Deterministic identity tuples make emission idempotent: re-runs (cache hits included) collide into verified no-ops instead of duplicating roots (the E13 hazard, relocated into graph creation and discharged).
- **`cjm_transcription_core.models`** — Data shapes for the transcription pipeline: run configuration + the run-manifest result containers. The run manifest is the pipeline's durable output record: which sources were processed, how they were segmented, and where each segment's transcription landed (capability data DBs remain the authoritative text store; the manifest records the run's shape + provenance pointers). It is a deliberate proto-bundle — the CR-20 provenance-bundle infrastructure is expected to absorb/replace it.
- **`cjm_transcription_core.pipeline`** — The headless transcription pipeline: VAD analysis -> boundary computation -> segment cutting -> per-segment model-input conversion -> transcription, composed over capability workers via the substrate's JobQueue. Between-stage outputs are threaded manually (run job -> read result -> submit next); the per-segment fan-out rides a CR-16 ports Composition with OutputRef bindings (this module was the real-world consumer of the original submit_sequence piping gap — pass-2 evidence in claude-docs/pass-2-evidence.md). HITL approval seams use the cheapest viable form (log + optional CLI prompt) per the cores-cluster guard-rails; each seam carries its 5-field HITL-assist annotation in its docstring.

## API

### `cjm_transcription_core.boundaries`

- `compute_segment_boundaries` _function_ — Group VAD chunks into segments cut at silence-gap midpoints.

### `cjm_transcription_core.cli`

- `build_parser` _function_ — Build the CLI parser (subcommands: run).
- `load_capabilities` _function_ — Discover manifests + load each requested capability (default instance).
- `main` _function_ — CLI entry point (console script: `cjm-transcription-core`).
- `parse_max_concurrent` _function_ — Parse repeatable `--max-concurrent NAME=N` values into a per-capability cap map.
- `run_command` _function_ — Execute the `run` subcommand: full pipeline over the given audio files.

### `cjm_transcription_core.emission`

- `build_source_emission` _function_ — Build the graph-root payload for one source (pure; no capability calls).
- `emit_source_graph` _function_ — Idempotently emit one source's graph root through the task channel.

### `cjm_transcription_core.models`

- `PipelineConfig` _class_ — Configuration for one transcription pipeline run.
- `RunManifest` _class_ — Durable record of one pipeline run (proto-bundle; see CR-20).
- `SegmentRecord` _class_ — One segment of a source audio file, with per-transcriber transcripts.
- `SourceResult` _class_ — Pipeline result for one source audio file.
- `new_run_id` _function_ — Generate a unique, sortable run id.

### `cjm_transcription_core.pipeline`

- `analyze_vad` _function_ — Run VAD analysis on one model-ready audio file (task channel: vad/detect_speech).
- `build_segment_composition` _function_ — Build the per-source fan-out composition: N independent [preprocess→]convert→(T× transcribe) pipes.
- `collect_capability_info` _function_ — Record capability identity + data-DB pointers for the run manifest (provenance).
- `confirm_seam` _function_ — HITL approval seam in its cheapest viable form (log + optional CLI prompt).
- `convert_for_vad` _function_ — Convert a source to MODEL-READY audio for VAD via the ffmpeg `convert` action.
- `cut_segments` _function_ — Cut the source audio at the computed boundaries via ffmpeg `segment_audio`.
- `normalize_vad_result` _function_ — Normalize a typed VAD result into sorted speech chunks + the reported duration.
- `probe_duration` _function_ — Probe a media file's duration via the ffmpeg capability's `get_info` action.
- `records_from_composition` _function_ — Fold a completed segment composition back into SegmentRecords.
- `run_pipeline` _function_ — Run the transcription pipeline over the given sources, in order.
- `run_source` _function_ — Run the full pipeline for one source: VAD → boundaries → cut → [preprocess →] convert → transcribe.
- `submit_and_wait` _function_ — Submit one capability job, wait for it, and return its result (raise on failure).
- `tier1_segment_checks` _function_ — Tier-1 deterministic pre-filters for the boundary-review seam (no AI).
- `tier1_transcript_checks` _function_ — Tier-1 deterministic pre-filters for the transcript-review seam (no AI).

## Dependencies

**Depends on:** `cjm-capability-primitives`, `cjm-context-graph-layer`, `cjm-context-graph-primitives`, `cjm-substrate`, `cjm-transcript-graph-schema`, `cjm-transcription-adapter-interface`
