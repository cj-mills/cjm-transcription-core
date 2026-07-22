"""Collection curation vocabulary (hub v0, e5849229): the journaled update/delete
ops the Collection layer needs beyond capture's pure ADDS. Capture (emission.py)
only ever extends; curation confirms a proposed collection, renames it (which is
MERGE when the new title already exists — title identity), refiles members, and
sets/repairs order. Every act flows through ONE journaled op shape (verb
`collection-curation`: deletes -> updates -> wires) with a domain-owned replay
handler (DEC 426658f1 posture), so the sidecar journal stays the source of truth
(ccbab9f5) and a rebuild replays curation exactly. TUIs never touch the graph
directly — the hub drives THIS vocabulary."""

import logging
from typing import Any, Dict, List, Optional, Tuple

from cjm_context_graph_layer.grammar import make_edge, spine_edges, SpineRelations
from cjm_context_graph_layer.identity import derive_edge_id
from cjm_context_graph_layer.journal import journal_extend
from cjm_context_graph_layer.ops import extend_graph, graph_task
from cjm_context_graph_primitives.journal import append_op
from cjm_context_graph_primitives.query import EdgeQuery, NodeQuery, RelationPredicate
from cjm_transcript_graph_schema.schema import CollectionNode, TranscriptGraphLabels

logger = logging.getLogger(__name__)


async def apply_curation(
    queue: Any,          # Started job queue
    graph_id: str,       # Graph-storage capability id
    op: Dict[str, Any],  # A journaled collection-curation op
) -> None:
    """Replay one `collection-curation` op: deletes -> updates -> wires.

    Every leg is idempotent (absent deletes are no-ops, updates merge, wires
    extend-verify), so replay onto ANY db state converges — the same guarantee
    `apply_wires` gives pure-add verbs, extended to the curation shape."""
    deletes = op.get("deletes") or {}
    if deletes.get("edge_ids"):
        await graph_task(queue, graph_id, "delete_edges",
                         edge_ids=list(deletes["edge_ids"]))
    if deletes.get("node_ids"):
        await graph_task(queue, graph_id, "delete_nodes",
                         node_ids=list(deletes["node_ids"]))
    for u in op.get("updates") or []:
        await graph_task(queue, graph_id, "update_node",
                         node_id=u["id"], properties=dict(u["properties"]))
    w = op.get("wires") or {}
    if w.get("nodes") or w.get("edges"):
        await extend_graph(queue, graph_id, w.get("nodes") or [], w.get("edges") or [])


def curation_replay_handlers() -> Dict[str, Any]:  # verb -> async handler(queue, graph_id, op)
    """The curation verb's replay registration (unioned into
    `transcription_replay_handlers` — collections were born at transcription
    hand-off, so their curation vocabulary replays under the same domain)."""
    return {"collection-curation": apply_curation}


async def journal_curation(
    queue: Any,                    # Started job queue
    graph_id: str,                 # Graph-storage capability id
    *,
    updates: Optional[List[Dict[str, Any]]] = None,   # [{"id", "properties"}] property merges
    delete_edge_ids: Optional[List[str]] = None,      # Edge ids to delete
    delete_node_ids: Optional[List[str]] = None,      # Node ids to delete (cascade)
    nodes: Optional[List[Dict[str, Any]]] = None,     # Node wires to extend with
    edges: Optional[List[Dict[str, Any]]] = None,     # Edge wires to extend with
    journal_path: Optional[str] = None,  # Sidecar journal (None = unjournaled)
    actor: str = "",               # Who curated (attribution)
    run: Optional[str] = None,     # Run id when a run drives it (usually None: curation is a human act)
    args: Optional[Dict[str, Any]] = None,  # Small semantic summary for the op
) -> Dict[str, Any]:  # The op as journaled (deletes/updates/wires)
    """Apply one curation act and journal it as a `collection-curation` op.

    The write-side dual of `apply_curation` — apply order matches replay order
    (deletes -> updates -> wires), and the op records the INTENT verbatim: unlike
    `journal_extend`'s added-delta trim, deletes and updates journal whole (they
    are already idempotent under replay, and trimming them would forget the act
    when the live db happened to be ahead)."""
    op: Dict[str, Any] = {"verb": "collection-curation", "actor": actor,
                          "args": dict(args or {}),
                          "deletes": {"edge_ids": list(delete_edge_ids or []),
                                      "node_ids": list(delete_node_ids or [])},
                          "updates": [dict(u) for u in (updates or [])],
                          "wires": {"nodes": list(nodes or []),
                                    "edges": list(edges or [])}}
    if run:
        op["run"] = run
    await apply_curation(queue, graph_id, op)
    if journal_path:
        append_op(journal_path, op, dedup=False)
    logger.info(f"collection-curation applied: {op['args']}")
    return op


async def list_collections(
    queue: Any,      # Started queue over the loaded graph capability
    graph_id: str,   # The graph capability name
) -> List[Dict[str, Any]]:  # [{"id", "title", "status"}] in query order
    """Enumerate the graph's Collection nodes (the hub's grouping corpus)."""
    cq = NodeQuery(label=TranscriptGraphLabels.COLLECTION, project=["title", "status"])
    res = await graph_task(queue, graph_id, "query_nodes", query=cq.to_dict())
    return [{"id": r["id"], "title": str(r.get("title") or ""),
             "status": str(r.get("status") or "proposed")}
            for r in (res.rows or [])]


async def collection_members(
    queue: Any,      # Started queue over the loaded graph capability
    graph_id: str,   # The graph capability name
    coll_id: str,    # Collection node id
) -> List[Tuple[str, str]]:  # [(source_id, title)] — membership, NOT order
    """A collection's member Sources (PART_OF edges; unordered by design —
    order, when it exists, is the NEXT chain `collection_order` walks)."""
    mq = NodeQuery(label=TranscriptGraphLabels.SOURCE,
                   related=RelationPredicate(SpineRelations.PART_OF, node_id=coll_id),
                   project=["title"])
    res = await graph_task(queue, graph_id, "query_nodes", query=mq.to_dict())
    return [(r["id"], str(r.get("title") or "")) for r in (res.rows or [])]


async def collection_order(
    queue: Any,      # Started queue over the loaded graph capability
    graph_id: str,   # The graph capability name
    coll_id: str,    # Collection node id
    member_ids: List[str],  # The collection's member Source ids (scope for the chain walk)
) -> Tuple[List[str], List[str]]:  # (ordered member ids from the chain walk; chain edge ids STARTS_WITH+NEXT)
    """Walk the materialized order, when one exists (typed EdgeQuery reads —
    the stage-4 endpoint-batch constraints were built for exactly this shape).

    Returns the members reachable from STARTS_WITH via NEXT, in order, plus the
    chain's edge ids (what `set_collection_order` deletes before re-chaining).
    No STARTS_WITH = unordered collection = ([], []); members a broken/partial
    chain cannot reach are simply absent from the walk (the unordered tail —
    a merge leaves exactly this shape until a reorder heals it)."""
    members = set(member_ids)
    sq = EdgeQuery(relation_type=SpineRelations.STARTS_WITH, source_id=coll_id,
                   project=["id"])
    sres = await graph_task(queue, graph_id, "query_edges", query=sq.to_dict())
    starts = [(r["id"], r["target_id"]) for r in (sres.rows or [])]
    if not starts:
        return [], []
    nxt: Dict[str, Tuple[str, str]] = {}
    if members:
        nq = EdgeQuery(relation_type=SpineRelations.NEXT,
                       source_ids=sorted(members), project=["id"])
        nres = await graph_task(queue, graph_id, "query_edges", query=nq.to_dict())
        nxt = {r["source_id"]: (r["id"], r["target_id"])
               for r in (nres.rows or []) if r["target_id"] in members}
    chain_edges = [e for e, _ in starts]  # tolerate stray STARTS_WITH
    ordered: List[str] = []
    cur = starts[0][1]
    while cur in members and cur not in ordered:
        ordered.append(cur)
        step = nxt.get(cur)
        if step is None:
            break
        chain_edges.append(step[0])
        cur = step[1]
    return ordered, chain_edges


async def file_sources(
    queue: Any,             # Started job queue
    graph_id: str,          # Graph-storage capability id
    title: str,             # Collection title (identity input; existing title = attach)
    member_ids: List[str],  # Source node ids to file (already on the graph)
    actor: str,             # Who filed (attribution)
    journal_path: Optional[str] = None,  # Sidecar journal
    status: str = "confirmed",  # A human filing in the hub IS a confirmation act
) -> Dict[str, Any]:  # {"collection_node_id", "nodes_added", "edges_added"}
    """File existing Sources into a collection (create-or-attach; the hub's
    late-binding path — ae3464fc: individual-items-first sources never wait).

    Pure ADDS, so it rides `journal_extend` under the capture verb
    `collection-declaration` (same replay lane as run-time capture). Title
    identity does the create-vs-attach resolution; membership lands UNORDERED
    (order is its own curation act)."""
    coll = CollectionNode(title=title, status=status, actor=actor)
    edges = [make_edge(m, coll.id, SpineRelations.PART_OF) for m in member_ids]
    res = await journal_extend(queue, graph_id, [coll.to_graph_node()], edges,
                               journal_path=journal_path, verb="collection-declaration",
                               actor=actor,
                               args={"collection_id": coll.id, "title": title,
                                     "status": status, "ordered": False,
                                     "curated": True})
    return {"collection_node_id": coll.id, "nodes_added": res.nodes_added,
            "edges_added": res.edges_added}


async def confirm_collection(
    queue: Any,          # Started job queue
    graph_id: str,       # Graph-storage capability id
    coll_id: str,        # Collection node id to confirm
    actor: str,          # The confirming human (attribution)
    journal_path: Optional[str] = None,  # Sidecar journal
) -> Dict[str, Any]:  # The journaled op
    """Discharge a proposed collection's flag (ae3464fc: the explicit human
    act the actor criterion waits for when capture ran hands-off)."""
    return await journal_curation(
        queue, graph_id,
        updates=[{"id": coll_id, "properties": {"status": "confirmed",
                                                "actor": actor}}],
        journal_path=journal_path, actor=actor,
        args={"act": "confirm", "collection_id": coll_id})


async def rename_collection(
    queue: Any,          # Started job queue
    graph_id: str,       # Graph-storage capability id
    coll_id: str,        # Collection node id to rename
    new_title: str,      # The new title — an EXISTING title makes this a MERGE
    actor: str,          # The renaming human (attribution)
    journal_path: Optional[str] = None,  # Sidecar journal
) -> str:  # The surviving collection's node id
    """Rename a collection — which IS merge when the new title already exists.

    Title is identity, so rename mints the new-title node (confirmed — renaming
    is a human act), re-files every member's PART_OF onto it, re-points
    STARTS_WITH when the old collection carried one, and deletes the old node
    (cascade cleans its remaining edges). When the new title's node already
    exists the same wires verify-collide and membership UNIONS — the d544e250
    guard surfaces that before the caller commits. NEXT edges live BETWEEN
    members and stay true under rename (same work, new name); a merge of two
    ordered collections keeps the survivor's STARTS_WITH and leaves the moved
    chain as an unordered tail until a reorder heals it."""
    members = await collection_members(queue, graph_id, coll_id)
    ordered, _ = await collection_order(queue, graph_id, coll_id,
                                        [m for m, _ in members])
    new = CollectionNode(title=new_title, status="confirmed", actor=actor)
    if new.id == coll_id:
        return coll_id  # normalization-equal title: nothing to move
    edges = [make_edge(m, new.id, SpineRelations.PART_OF) for m, _ in members]
    if ordered:
        edges.append(make_edge(new.id, ordered[0], SpineRelations.STARTS_WITH))
    await journal_curation(
        queue, graph_id,
        delete_node_ids=[coll_id],
        nodes=[new.to_graph_node()], edges=edges,
        journal_path=journal_path, actor=actor,
        args={"act": "rename", "from": coll_id, "to": new.id, "title": new_title})
    return new.id


async def refile_members(
    queue: Any,              # Started job queue
    graph_id: str,           # Graph-storage capability id
    member_ids: List[str],   # Source ids to move
    from_coll_id: str,       # Collection they leave
    to_title: str,           # Collection they join (create-or-attach by title)
    actor: str,              # The refiling human (attribution)
    journal_path: Optional[str] = None,  # Sidecar journal
) -> str:  # The destination collection's node id
    """Move members between collections (the Supernova carve-out: select
    episodes out of a coarse archive proposal into their true series).

    Deterministic edge ids make the deletes recomputable — no lookup: the old
    PART_OF (and a defensive STARTS_WITH, if a moved member anchored the old
    chain) delete by derived id. Destination is create-or-attach by title
    (confirmed — refiling is a human act); moved members land UNORDERED there.
    NEXT edges among moved members persist (still true of the work itself);
    the source collection's chain heals on its next reorder."""
    dst = CollectionNode(title=to_title, status="confirmed", actor=actor)
    dels = [derive_edge_id(m, from_coll_id, SpineRelations.PART_OF) for m in member_ids]
    dels += [derive_edge_id(from_coll_id, m, SpineRelations.STARTS_WITH) for m in member_ids]
    edges = [make_edge(m, dst.id, SpineRelations.PART_OF) for m in member_ids]
    await journal_curation(
        queue, graph_id,
        delete_edge_ids=dels,
        nodes=[dst.to_graph_node()], edges=edges,
        journal_path=journal_path, actor=actor,
        args={"act": "refile", "from": from_coll_id, "to": dst.id,
              "title": to_title, "members": len(member_ids)})
    return dst.id


async def set_collection_order(
    queue: Any,                    # Started job queue
    graph_id: str,                 # Graph-storage capability id
    coll_id: str,                  # Collection node id
    ordered_member_ids: List[str],  # The full member order to materialize
    actor: str,                    # The ordering human (attribution)
    journal_path: Optional[str] = None,  # Sidecar journal
) -> Dict[str, Any]:  # The journaled op
    """Materialize (or repair) a collection's order — the curation op ae3464fc
    reserved sequence for.

    Deletes the existing chain (STARTS_WITH + reachable NEXT via
    `collection_order`, PLUS any stray NEXT edges among the members — the
    unordered tails a merge leaves), then lays the full fractal spine over the
    given order (`spine_edges`; the PART_OF re-adds verify-collide)."""
    members = set(ordered_member_ids)
    _, chain_edges = await collection_order(queue, graph_id, coll_id,
                                            list(members))
    stray: List[str] = []
    if members:
        nq = EdgeQuery(relation_type=SpineRelations.NEXT,
                       source_ids=sorted(members), project=["id"])
        nres = await graph_task(queue, graph_id, "query_edges", query=nq.to_dict())
        stray = [r["id"] for r in (nres.rows or [])]
    return await journal_curation(
        queue, graph_id,
        delete_edge_ids=sorted(set(chain_edges) | set(stray)),
        edges=spine_edges(coll_id, list(ordered_member_ids)),
        journal_path=journal_path, actor=actor,
        args={"act": "set-order", "collection_id": coll_id,
              "members": len(ordered_member_ids)})
