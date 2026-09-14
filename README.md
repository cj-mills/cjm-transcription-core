# cjm-transcription-core

<!-- generated from the context graph by `cjm-context-graph readme` — do not edit by hand; edit the graph (the urge to hand-edit = move it on-graph) -->

A frontend-agnostic core for the audio transcription workflow — composes isolated capability workers (audio conversion, VAD segmentation, batch transcription, persistence) into a headless pipeline, with a CLI as its first driver.

## Modules

- **`cjm_transcription_core.__init__`**
- **`cjm_transcription_core.boundaries`** — Wall-clock-aware segment boundary computation: group VAD speech chunks into segments cut at silence-gap midpoints. Pure logic — no capability calls. Final home of the algorithm originally validated in cjm-transcription-audio-segment's AudioSegmentService.compute_segment_boundaries (that library is retired to cj-mills_deferred/).
- **`cjm_transcription_core.candidates`** — Candidate (capability, MODEL)-instance enumeration for the comparison screen.
- **`cjm_transcription_core.chunk`** — Chunk-grain re-runs, external landings and the runaway census (work item cf0b91d6; rulings 9ffce5f7 · 8a9b9639 · 910f3692): ONE landing, two producers — a local transcriber re-run of one AudioSegment's rendition, or an operator-pasted external transcript — lands a Transcript variant under the EXISTING AudioSegment with provenance, SUPERSEDES the prior variant for that (rendition, transcriber), journals the delta, and writes a DERIVED run manifest (parent_run_id + per-entry config_hash) the decomp consumes; the census names the chunks in need (degenerate-tail markers on new runs, oversized / implausible words-per-second text on old ones, extreme two-transcriber disagreement) and never counts a superseded variant as live.
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

### `cjm_transcription_core.chunk`

- `apply_chunk_update` _function_ — Replace ONE chunk's entry for ONE transcriber in the derived manifest.
- `build_chunk_landing` _function_ — Build the landing payload for ONE chunk (pure; no capability calls).
- `census_rows` _function_ — The runaway census (pure): which LIVE chunk variants need a better transcription.
- `chunks_from_census` _function_ — Map census rows onto the manifest's chunks by (Source id, segment index) — the
- `derive_manifest` _function_ — Start a DERIVED manifest (ruling 910f3692 (1)): a copy of the parent under a NEW
- `fetch_transcript_rows` _function_ — Pull every Transcript's census inputs through the graph capability's marked
- `flagged_chunks` _function_ — The inspection lane's jump index: which chunks of a run carry a flagged variant,
- `is_external_transcriber` _function_ — Whether a transcriber name is an external landing's (the `/manual` marker).
- `land_chunk_transcript` _function_ — Land one chunk's variant through the task channel and journal the delta —
- `load_run_manifest` _function_ — Load a transcription-core run manifest (${WS}/ recorded paths resolve at load,
- `prior_config_hash` _function_ — The config hash of the variant this transcriber currently has on the chunk: a
- `prompt_hash_of` _function_ — Hash a prompt template — the prompt is DATA (f304d31d) and its hash rides the
- `render_escalation_prompt` _function_ — Render the escalation prompt WITH CONTEXT for one chunk (ruling 8a9b9639 (3):
- `rows_from_manifest` _function_ — Census inputs from a run manifest alone (no graph): the qt inspection lane's
- `save_manifest` _function_ — Write the derived manifest (the same recording contract as RunManifest.save).
- `select_chunks` _function_ — Resolve a chunk selection against the manifest (pure).
- `summarize_census` _function_ — Per-collection roll-up of the census (the closing evidence for 56a802b3 is a
- `text_shape` _function_ — Census the newline shape of a pasted external transcript (finding efe88f17;
- `wordwrap_warning` _function_ — The landing-time warning for a wordwrap-shaped paste (efe88f17 (2)): the

### `cjm_transcription_core.cli`

- `add_transcript_command` _function_ — Execute `add-transcript` (cf0b91d6 part 2; ruling 9ffce5f7 (1)): land an
- `build_parser` _function_ — Build the CLI parser (subcommands: run).
- `declare_structure_command` _function_ — Execute `declare-structure`: read a structure-map document and land it
- `expand_sources` _function_ — Expand CLI source arguments into the ordered media-file list for a run.
- `expand_sources_with_collections` _function_ — Expand CLI sources AND keep the folder-source gesture as collection
- `load_capabilities` _function_ — Discover manifests + load each requested capability.
- `main` _function_ — CLI entry point (console script: `cjm-transcription-core`).
- `parse_config_overrides` _function_ — Parse repeatable KEY=VALUE config overrides (`--transcriber-config`).
- `parse_max_concurrent` _function_ — Parse repeatable `--max-concurrent NAME=N` values into a per-capability cap map.
- `parse_transcriber_spec` _function_ — Parse one `--transcriber` spec into a (capability, MODEL)-instance load directive.
- `reference_command` _function_ — Execute `add-reference` / `retract-reference`: attach or retract a human-added
- `rerun_chunk_command` _function_ — Execute `rerun-chunk` (cf0b91d6 part 1; ruling 8a9b9639 chunk-targeted, never wholesale).
- `retire_collection_command` _function_ — Execute `retire-collection` (ruling a7617bd4, item eaefebd2): resolve the
- `run_command` _function_ — Execute the `run` subcommand: full pipeline over the given audio files.
- `runaway_census_command` _function_ — Execute `runaway-census` (cf0b91d6 part 3): the LIVE chunk variants in need of a

### `cjm_transcription_core.curation`

- `add_reference` _function_ — Attach a HUMAN-ADDED RESOURCE LINK to a Source as a `Reference` NODE (ruling
- `apply_curation` _function_ — Replay one `collection-curation` op: deletes -> updates -> wires.
- `collection_members` _function_ — A collection's member Sources (PART_OF edges; unordered by design —
- `collection_order` _function_ — Walk the materialized order, when one exists (typed EdgeQuery reads —
- `confirm_collection` _function_ — Discharge a proposed collection's flag (ae3464fc: the explicit human
- `curation_replay_handlers` _function_ — The curation verb's replay registration (unioned into
- `declare_structure` _function_ — Declare a SOURCE STRUCTURE MAP: the WORK's own part/chapter structure
- `file_sources` _function_ — File existing Sources into a collection (create-or-attach; the hub's
- `journal_curation` _function_ — Apply one curation act and journal it as a `collection-curation` op.
- `list_collections` _function_ — Enumerate the graph's Collection nodes (the hub's grouping corpus).
- `live_collections` _function_ — Filter retired collections out of a listing (pure; the pickers' default view).
- `refile_members` _function_ — Move members between collections (the Supernova carve-out: select
- `rename_collection` _function_ — Rename a collection — which IS merge when the new title already exists.
- `retire_collection` _function_ — Retire a Collection as a journaled FACT (ruling a7617bd4, item eaefebd2): status
- `retract_reference` _function_ — Retract a `Reference` node — the compensating act for `add_reference` (the node
- `set_collection_order` _function_ — Materialize (or repair) a collection's order — the curation op ae3464fc
- `structure_entries_from_map` _function_ — Normalize a structure-map document into `declare_structure` entries.

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
