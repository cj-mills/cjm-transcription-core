"""Graph-root emission (CR-18 revolution 2): a completed source EMITS Source -> AudioSegment -> Transcript into the shared context graph — the graph BEGINS at transcription (where-graph-begins resolution: ingestion is the first EXTENDER that plants the root). Deterministic identity tuples make emission idempotent: re-runs (cache hits included) collide into verified no-ops instead of duplicating roots (the E13 hazard, relocated into graph creation and discharged)."""

import logging
from typing import Any, Dict, List, Optional, Tuple

from cjm_context_graph_layer.declare import Derivation, derivation_to_graph
from cjm_context_graph_layer.grammar import spine_edges
from cjm_context_graph_layer.journal import journal_extend, wires_handlers
from cjm_substrate.core.queue import JobQueue
from cjm_transcript_graph_schema.schema import (AudioRenditionNode, AudioSegmentNode, SourceNode,
                                                TranscriptNode)
from cjm_transcription_core.models import SourceResult

logger = logging.getLogger(__name__)


def build_source_emission(
    src: SourceResult,                          # Completed per-source pipeline result (0.3.0 shape)
    transcriber_config_hashes: Dict[str, str],  # transcriber -> effective config hash (Transcript identity input)
    chain: Optional[List[str]] = None,          # Preprocessing chain that produced the model-inputs ([]/None = raw convert-only)
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:  # (nodes, edges, ids)
    """Build the graph-root payload for one source (pure; no capability calls).

    Emits the locked layer schema: one `Source` (identity = file content hash)
    -> coarse `AudioSegment` boundary spine (PART_OF / NEXT / STARTS_WITH via
    `spine_edges`) -> one `AudioRendition` per segment (the model-input WAV; it
    DERIVED_FROM its AudioSegment) -> per-transcriber `Transcript` variants
    (DERIVED_FROM their rendition; identity mirrors the capability cache key).
    Returns the ids dict {"source", "audio_segments", "renditions",
    "transcripts"} for callers (decomp recomputes these same ids from the
    manifest — no stored-id coupling).

    `chain` is the AudioRendition IDENTITY input: an empty chain is the raw
    convert-only rendition; a non-empty chain (e.g. demucs vocals) yields a
    DISTINCT rendition node under the SAME AudioSegment, so raw + preprocessed
    model-inputs of one boundary COEXIST in one graph (the divergent
    model_input_hash now lives on distinct rendition nodes, not a colliding
    AudioSegment). The chain rides into the node id, so re-derivation reproduces
    it from the manifest alone.
    """
    if not src.content_hash:
        raise ValueError(f"source {src.source_path} has no content_hash — emission identity requires it")
    chain = list(chain or [])
    source = SourceNode(content_hash=src.content_hash, path=src.source_path)
    nodes: List[Dict[str, Any]] = [source.to_graph_node()]
    edges: List[Dict[str, Any]] = []
    aseg_ids: List[str] = []
    rendition_ids: List[str] = []
    transcript_ids: Dict[str, List[str]] = {}

    for rec in src.segments:
        if not rec.model_input_hash:
            raise ValueError(f"segment {rec.index} has no model_input_hash — emission identity requires it")
        aseg = AudioSegmentNode(
            source=source.id, index=rec.index, start=rec.start, end=rec.end,
            segment_path=rec.segment_path,
        )
        nodes.append(aseg.to_graph_node())
        aseg_ids.append(aseg.id)
        # The model-input WAV is the rendition's, not the boundary's.
        rendition = AudioRenditionNode(
            audio_segment=aseg.id, model_input_path=rec.model_input_path,
            model_input_hash=rec.model_input_hash, chain=chain,
        )
        nodes.append(rendition.to_graph_node())
        edges.append(rendition.derived_edge())  # rendition DERIVED_FROM its AudioSegment
        rendition_ids.append(rendition.id)
        for tname, tr in rec.transcripts.items():
            tnode = TranscriptNode(
                rendition=rendition.id, transcriber=tname,
                config_hash=transcriber_config_hashes.get(tname, ""),
                text=str(tr.get("text") or ""), audio_hash=rec.model_input_hash,
                metadata=dict(tr.get("metadata") or {}),
            )
            nodes.append(tnode.to_graph_node())
            edges.append(tnode.derived_edge())  # transcript DERIVED_FROM its rendition
            transcript_ids.setdefault(tname, []).append(tnode.id)

    edges = spine_edges(source.id, aseg_ids) + edges
    ids = {"source": source.id, "audio_segments": aseg_ids,
           "renditions": rendition_ids, "transcripts": transcript_ids}
    return nodes, edges, ids


async def emit_source_graph(
    queue: JobQueue,                            # Started job queue
    graph_id: str,                              # Graph-storage capability instance id
    src: SourceResult,                          # Completed per-source pipeline result
    transcriber_config_hashes: Dict[str, str],  # transcriber -> effective config hash
    run_id: str,                                # Run id (recorded on the boundary Derivation event)
    chain: Optional[List[str]] = None,          # Preprocessing chain that produced the model-inputs ([]/None = raw)
    journal_path: Optional[str] = None,         # Sidecar write journal — DELTA ops append on success (None = unjournaled)
) -> Dict[str, Any]:  # Emission record for the manifest
    """Idempotently emit one source's graph root through the task channel.

    `extend_graph` = emit-if-absent + verify-if-present, so a re-run over
    cached content collides into a verified no-op (stress item 4) and a second
    transcriber's run EXTENDS the existing root (only its Transcript nodes are
    new). The host's contribution this run — boundary computation and/or the
    preprocessing chain that produced new renditions — is declared as a
    `Derivation` event (provenance-by-declaration) ONLY when it actually created
    AudioSegment or AudioRendition nodes (a preprocessing-ON run into a graph
    that already holds the raw boundaries creates new renditions but no new
    boundaries — both cases declare; verified re-emissions don't spam the audit
    trail).
    """
    chain = list(chain or [])
    nodes, edges, ids = build_source_emission(src, transcriber_config_hashes, chain=chain)
    res = await journal_extend(queue, graph_id, nodes, edges,
                               journal_path=journal_path, verb="source-emission",
                               actor="pipeline:cjm-transcription-core", run=run_id,
                               args={"source_id": ids["source"], "chain": chain})
    added = set(res.added_node_ids)
    new_asegs = [a for a in ids["audio_segments"] if a in added]
    new_renditions = [r for r in ids["renditions"] if r in added]
    if new_asegs or new_renditions:
        parts = (["segment-boundaries"] if new_asegs else []) + (["preprocessing"] if (chain and new_renditions) else [])
        method = ("+".join(parts) or "renditions") + "/v1"
        props: Dict[str, Any] = {"run_id": run_id}
        if chain:
            props["chain"] = list(chain)
        d = Derivation(
            actor="host:cjm-transcription-core", method=method,
            input_ids=[ids["source"]], output_ids=new_asegs + new_renditions,
            properties=props,
        )
        dn, de = derivation_to_graph(d)
        await journal_extend(queue, graph_id, [dn], de,
                             journal_path=journal_path, verb="derivation",
                             actor="pipeline:cjm-transcription-core", run=run_id,
                             args={"method": method})
    record = {
        "source_node_id": ids["source"],
        "nodes_added": res.nodes_added,
        "nodes_verified": res.nodes_verified,
        "edges_added": res.edges_added,
        "edges_existing": res.edges_existing,
    }
    logger.info(f"emitted {src.source_path}: {record}")
    return record


def transcription_replay_handlers() -> Dict[str, Any]:  # verb -> async handler(queue, graph_id, op)
    """The transcription core's replay vocabulary (DEC 426658f1, replay stays DOMAIN-OWNED).

    Exported through the `cjm_context_graph_layer.replay` entry-point group and
    unioned by `composed_replay_handlers`. Both verbs are `journal_extend` ops —
    wire-carrying by construction — so they register the layer's shared
    `apply_wires` (identity-comparable across cores: decomp also emits
    `derivation`, and the shared handler keeps that collision legal)."""
    return wires_handlers("source-emission", "derivation")
