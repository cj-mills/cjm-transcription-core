# cjm-transcription-core

<!-- generated from the context graph by `cjm-context-graph readme` — do not edit by hand; edit the graph (the urge to hand-edit = move it on-graph) -->

A frontend-agnostic core for the audio transcription workflow — composes isolated capability workers (audio conversion, VAD segmentation, batch transcription, persistence) into a headless pipeline, with a CLI as its first driver.

## Modules

- **`cjm_transcription_core`**
- **`cjm_transcription_core.boundaries`** — Wall-clock-aware segment boundary computation: group VAD speech chunks into segments cut at silence-gap midpoints. Pure logic — no capability calls. Final home of the algorithm originally validated in cjm-transcription-audio-segment's AudioSegmentService.compute_segment_boundaries (that library is retired to cj-mills_deferred/).
- **`cjm_transcription_core.candidates`** — Candidate (capability, MODEL)-instance enumeration for the comparison screen.
- **`cjm_transcription_core.cli`** — The CLI driver — the workflow core's first (and currently only) frontend.
- **`cjm_transcription_core.curation`** — Collection curation vocabulary (hub v0, e5849229): the journaled update/delete
- **`cjm_transcription_core.emission`** — Graph-root emission (CR-18 revolution 2): a completed source EMITS Source -> AudioSegment -> Transcript into the shared context graph — the graph BEGINS at transcription (where-graph-begins resolution: ingestion is the first EXTENDER that plants the root). Deterministic identity tuples make emission idempotent: re-runs (cache hits included) collide into verified no-ops instead of duplicating roots (the E13 hazard, relocated into graph creation and discharged).
- **`cjm_transcription_core.launch`** — The shared launch surface every transcription shell drives through: the
- **`cjm_transcription_core.models`** — Data shapes for the transcription pipeline: run configuration + the run-manifest result containers. The run manifest is the pipeline's durable output record: which sources were processed, how they were segmented, and where each segment's transcription landed (capability data DBs remain the authoritative text store; the manifest records the run's shape + provenance pointers). It is a deliberate proto-bundle — the CR-20 provenance-bundle infrastructure is expected to absorb/replace it.
- **`cjm_transcription_core.pipeline`** — The headless transcription pipeline: VAD analysis -> boundary computation -> segment cutting -> per-segment model-input conversion -> transcription, composed over capability workers via the substrate's JobQueue. Between-stage outputs are threaded manually (run job -> read result -> submit next); the per-segment fan-out rides a CR-16 ports Composition with OutputRef bindings (this module was the real-world consumer of the original submit_sequence piping gap — pass-2 evidence in claude-docs/pass-2-evidence.md). HITL approval seams use the cheapest viable form (log + optional CLI prompt) per the cores-cluster guard-rails; each seam carries its 5-field HITL-assist annotation in its docstring.
- **`cjm_transcription_core.probe`** — Per-segment comparison probe: transcribe ONE VAD-cut segment across every
- **`cjm_transcription_core.results`** — Past-run results for the setup TUI: the core's own runs/*.json manifests read
- **`cjm_transcription_core.sources`** — Source-selection state for the picker stage: a keyboard file browser plus the
- **`cjm_transcription_core.state`** — Sidecar TUI state: last-used run settings persisted across sessions (the

## API

### `cjm_transcription_core.boundaries`

- `compute_segment_boundaries` _function_ — Group VAD chunks into segments cut at silence-gap midpoints.

### `cjm_transcription_core.candidates`

- `candidate_directives` _function_ — Expand every installed transcription capability into its candidate space.
- `discover_capability` _function_ — Pick a DEFAULT capability for a role by surface match.
- `instance_id_for` _function_ — Derive an addressable instance id for a non-default (capability, MODEL) pick.
- `manifests_with_method` _function_ — Enumerate installed capabilities whose structural surface lists `method`.
- `model_axis` _function_ — Find a capability's MODEL config axis in its config_schema.
- `spec_string` _function_ — Render a load directive back to the core CLI's --transcriber grammar.
- `transcription_manifests` _function_ — Enumerate installed transcription capabilities from their manifest files.

### `cjm_transcription_core.cli`

- `build_parser` _function_ — Build the CLI parser (subcommands: run).
- `expand_sources` _function_ — Expand CLI source arguments into the ordered media-file list for a run.
- `expand_sources_with_collections` _function_ — Expand CLI sources AND keep the folder-source gesture as collection
- `load_capabilities` _function_ — Discover manifests + load each requested capability.
- `main` _function_ — CLI entry point (console script: `cjm-transcription-core`).
- `parse_max_concurrent` _function_ — Parse repeatable `--max-concurrent NAME=N` values into a per-capability cap map.
- `parse_transcriber_spec` _function_ — Parse one `--transcriber` spec into a (capability, MODEL)-instance load directive.
- `run_command` _function_ — Execute the `run` subcommand: full pipeline over the given audio files.

### `cjm_transcription_core.curation`

- `apply_curation` _function_ — Replay one `collection-curation` op: deletes -> updates -> wires.
- `collection_members` _function_ — A collection's member Sources (PART_OF edges; unordered by design —
- `collection_order` _function_ — Walk the materialized order, when one exists (typed EdgeQuery reads —
- `confirm_collection` _function_ — Discharge a proposed collection's flag (ae3464fc: the explicit human
- `curation_replay_handlers` _function_ — The curation verb's replay registration (unioned into
- `file_sources` _function_ — File existing Sources into a collection (create-or-attach; the hub's
- `journal_curation` _function_ — Apply one curation act and journal it as a `collection-curation` op.
- `list_collections` _function_ — Enumerate the graph's Collection nodes (the hub's grouping corpus).
- `refile_members` _function_ — Move members between collections (the Supernova carve-out: select
- `rename_collection` _function_ — Rename a collection — which IS merge when the new title already exists.
- `set_collection_order` _function_ — Materialize (or repair) a collection's order — the curation op ae3464fc

### `cjm_transcription_core.emission`

- `build_collection_emission` _function_ — Build the Collection layer payload for one declaration (pure; no
- `build_source_emission` _function_ — Build the graph-root payload for one source (pure; no capability calls).
- `emit_collections_graph` _function_ — Idempotently emit the run's collection declarations (verb
- `emit_source_graph` _function_ — Idempotently emit one source's graph root through the task channel.
- `transcription_replay_handlers` _function_ — The transcription core's replay vocabulary (DEC 426658f1, replay stays DOMAIN-OWNED).

### `cjm_transcription_core.launch`

- `build_parser` _function_ — The TUI driver's argument surface (setup options + core-run passthrough).
- `hand_off` _function_ — The shared driver tail: persist the confirmed choices, print the
- `plan_argv` _function_ — Render a confirmed plan as headless core-CLI argv.
- `resolve_settings` _function_ — Resolve the run-setup settings every shell shares (flags > persisted

### `cjm_transcription_core.models`

- `CollectionDecl` _class_ — A collection declaration riding a run (ae3464fc: the folder-source
- `PipelineConfig` _class_ — Configuration for one transcription pipeline run.
- `RunManifest` _class_ — Durable record of one pipeline run (proto-bundle; see CR-20).
- `SegmentRecord` _class_ — One segment of a source audio file, with per-transcriber transcripts.
- `SourceResult` _class_ — Pipeline result for one source audio file.
- `new_run_id` _function_ — Generate a unique, sortable run id.

### `cjm_transcription_core.pipeline`

- `acquire_speaker_turns` _function_ — Diarize the full source and persist the source-keyed turns artifact.
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

### `cjm_transcription_core.probe`

- `SegmentProbe` _class_ — One source's cut segments + cached per-segment comparison results.

### `cjm_transcription_core.results`

- `RunIndex` _class_ — runs/*.json manifests loaded newest-first + the lookups the TUI paints from.

### `cjm_transcription_core.sources`

- `CollectionField` _class_ — Pre-run collection state for the sources stage (ae3464fc: the actor
- `SourceBrowser` _class_ — Keyboard file-browser + ordered selection state for the sources stage.

### `cjm_transcription_core.state`

- `load_state` _function_ — Read this project's persisted TUI state.
- `save_state` _function_ — Merge updates into the persisted state and write it back (best-effort:
- `state_path` _function_ — Where this project's TUI state lives.

## Dependencies

**Depends on:** `cjm-capability-primitives`, `cjm-context-graph-layer`, `cjm-context-graph-primitives`, `cjm-substrate`, `cjm-transcript-graph-schema`, `cjm-transcription-adapter-interface`
**Used by:** `cjm-transcript-correction-qt`, `cjm-transcription-qt`, `cjm-workflow-hub-qt`
