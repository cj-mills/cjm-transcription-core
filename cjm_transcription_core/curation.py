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
import re
import time
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

from cjm_context_graph_layer.grammar import make_edge, spine_edges, SpineRelations
from cjm_context_graph_layer.identity import derive_edge_id, derive_node_id
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


async def holding_collections(
    queue: Any,      # Started queue over the loaded graph capability
    graph_id: str,   # The graph capability name
    source_id: str,  # Source node id
) -> List[Dict[str, Any]]:  # [{"id", "title", "status"}] — every Collection the Source is PART_OF
    """The Collections holding a Source — the inverse of `collection_members`,
    read from the Source side (PART_OF edges INTO the candidate Collection).
    One query, no corpus sweep: the correction app's direct `--source` open
    resolves a source's grouping without the browse ladder (DEC 774dbe40)."""
    cq = NodeQuery(label=TranscriptGraphLabels.COLLECTION,
                   related=RelationPredicate(SpineRelations.PART_OF, direction="in",
                                             node_id=source_id),
                   project=["title", "status"])
    res = await graph_task(queue, graph_id, "query_nodes", query=cq.to_dict())
    return [{"id": r["id"], "title": str(r.get("title") or ""),
             "status": str(r.get("status") or "proposed")}
            for r in (res.rows or [])]


async def sibling_sources(
    queue: Any,      # Started queue over the loaded graph capability
    graph_id: str,   # The graph capability name
    source_id: str,  # Source node id
) -> Dict[str, Any]:  # {"collections": [{"id","title","status"}] (live only), "siblings": {source_id: title}} — the source itself excluded
    """A Source's collection siblings: the members of every LIVE collection
    holding it (a retired collection is hidden with its members — ruling
    a7617bd4 — so its speakers never crowd a picker), unioned across holding
    collections, the source itself left out. The assign-lane picker's
    collection tier reads exactly this set (DEC 774dbe40)."""
    live = [c for c in await holding_collections(queue, graph_id, source_id)
            if c["status"] != "retired"]
    siblings: Dict[str, str] = {}
    for c in live:
        for sid, title in await collection_members(queue, graph_id, c["id"]):
            if sid != source_id:
                siblings.setdefault(sid, title)
    return {"collections": live, "siblings": siblings}


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


async def retire_collection(
    queue: Any,          # Started job queue
    graph_id: str,       # Graph-storage capability id
    coll_id: str,        # Collection node id to retire (or restore)
    actor: str,          # The retiring human (attribution)
    reason: str = "",    # Why (journaled; e.g. re-downloaded as a fresh collection)
    journal_path: Optional[str] = None,  # Sidecar journal
    unretire: bool = False,  # Restore a retired collection to confirmed
) -> Dict[str, Any]:  # The journaled op
    """Retire a Collection as a journaled FACT (ruling a7617bd4, item eaefebd2): status
    becomes `retired` and pickers / the census skip it; its Sources, spines and
    corrections stay on the graph untouched (a retired collection is hidden, never
    cascaded). Reversible with `unretire` (status back to confirmed)."""
    props: Dict[str, Any] = ({"status": "confirmed", "retired": None} if unretire else
                             {"status": "retired",
                              "retired": {"reason": reason or "", "actor": actor, "ts": time.time()}})
    return await journal_curation(
        queue, graph_id,
        updates=[{"id": coll_id, "properties": props}],
        journal_path=journal_path, actor=actor,
        args={"act": "unretire-collection" if unretire else "retire-collection",
              "collection_id": coll_id, "reason": reason or ""})


def live_collections(
    collections: List[Dict[str, Any]],  # list_collections rows
) -> List[Dict[str, Any]]:  # The rows whose status is not `retired`
    """Filter retired collections out of a listing (pure; the pickers' default view)."""
    return [c for c in collections if str(c.get("status") or "") != "retired"]


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


def structure_entries_from_map(
    doc: Dict[str, Any],  # A structure-map document: {"collection_id"?, "entries": [{"source_id", "evidence"?, ...cells}]}
) -> List[Dict[str, Any]]:  # [{"source_id", "structure": {cells}, "evidence": [...]}] in document order
    """Normalize a structure-map document into `declare_structure` entries.

    The document is the human-facing shape (one row per member Source with
    the work's cells laid flat: file / kind / part / part_title / chapter /
    unit / title); the entry is the op-facing shape (cells folded under
    `structure`, `evidence` beside them). Keys prefixed `_` are document
    commentary and never land; `source_id` and `evidence` are lifted out;
    every remaining key is a cell. A row without a `source_id` or without any
    cell refuses loudly — a map that names nothing declares nothing."""
    entries: List[Dict[str, Any]] = []
    for row in doc.get("entries") or []:
        sid = row.get("source_id")
        if not sid:
            raise ValueError(f"structure map row without source_id: {row!r}")
        cells = {k: v for k, v in row.items()
                 if k not in ("source_id", "evidence") and not str(k).startswith("_")}
        if not cells:
            raise ValueError(f"structure map row {sid!r} carries no cells")
        entries.append({"source_id": sid, "structure": cells,
                        "evidence": list(row.get("evidence") or [])})
    return entries


def url_bindings_from_playlist(
    members: List[Dict[str, Any]],   # The collection's member Sources: [{"id", "title"}]
    items: List[Dict[str, Any]],     # Playlist metadata rows: [{"title", "url", ...}] (youtube-playlist-extractor.py output)
    *,
    playlist_file: Optional[str] = None,  # Where the rows came from (rides each binding as evidence)
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:  # (bindings, unmatched members)
    """Pure: join a collection's member Sources to a playlist's rows BY TITLE and return
    one binding per matched Source plus the members no row matched (ruling f0b7f125 (4):
    the playlist metadata JSON is the public-URL authority for a YouTube-sourced
    collection). A downloaded video's file name — hence the Source title — carries the
    characters YouTube's title cannot spell on disk (a fullwidth colon '：' for ':', '⧸'
    for '/'), so both sides fold through NFKC + the slash swap + whitespace/case before
    comparing. Second tier (live sighting: three GPU MODE streams re-titled on YouTube
    after download): when no exact key matches, the 'Lecture N: ' prefix is stripped
    from both sides and a UNIQUE bare-title match binds; a bare title two rows share
    stays unmatched. A member that matches nothing is REPORTED, never guessed: the
    human re-titles or binds it by hand."""
    def _key(title: Any) -> str:
        t = unicodedata.normalize("NFKC", str(title or ""))
        t = t.replace("⧸", "/").replace("／", "/")
        return " ".join(t.split()).strip().lower()

    by_key: Dict[str, Dict[str, Any]] = {}
    for it in items:
        k = _key(it.get("title"))
        if k and k not in by_key and it.get("url"):
            by_key[k] = it
    bindings: List[Dict[str, Any]] = []
    unmatched: List[Dict[str, Any]] = []
    def _bare(k: str) -> str:  # second tier: the 'Lecture N: ' prefix a re-titled video gained or lost
        return re.sub(r"^lecture \d+[a-z]?:\s*", "", k)

    by_bare: Dict[str, List[Dict[str, Any]]] = {}
    for k, it in by_key.items():
        by_bare.setdefault(_bare(k), []).append(it)
    for m in members:
        k = _key(m.get("title"))
        hit = by_key.get(k)
        if hit is None:
            cands = by_bare.get(_bare(k)) or []
            hit = cands[0] if len(cands) == 1 else None   # a bare title shared by two rows stays unmatched
        if hit is None:
            unmatched.append({"source_id": m["id"], "title": m.get("title")})
            continue
        bindings.append({"source_id": m["id"], "title": m.get("title"), "url": str(hit["url"]).strip(),
                         "playlist_title": hit.get("title"),
                         **({"playlist_file": playlist_file} if playlist_file else {})})
    return bindings, unmatched


async def bind_source_urls(
    queue: Any,                            # Started job queue
    graph_id: str,                         # Graph-storage capability id
    bindings: List[Dict[str, Any]],        # [{"source_id", "url", "playlist_title"?, "playlist_file"?}] — one per Source
    actor: str,                            # The binding human (attribution; the playlist file is human-generated data)
    journal_path: Optional[str] = None,    # Sidecar journal
    collection_id: Optional[str] = None,   # The Collection the bindings lie over (semantic summary only)
) -> Dict[str, Any]:  # The journaled op
    """Bind each Source's PUBLIC URL — the time-addressable watch page a rendering links
    spans into (`read_source_unit` reads `public_url`; ruling e1fd4d64 (D): only an
    addressable source renders timestamps, as links). Lands as a `public_url` property
    merge per Source plus a `public_url_evidence` map saying where the URL came from
    (the playlist metadata file + the row title that matched) — the declare-structure
    shape: a PROPERTY merge riding `journal_curation` (verb `collection-curation`, act
    `bind-source-urls`), idempotent under replay, a re-bind simply overwrites. A binding
    without a URL refuses before anything is applied."""
    updates: List[Dict[str, Any]] = []
    for b in bindings:
        sid = b["source_id"]
        url = str(b.get("url") or "").strip()
        if not url:
            raise ValueError(f"bind_source_urls: binding {sid!r} carries no url")
        evidence: Dict[str, Any] = {"kind": "playlist-metadata"}
        for k in ("playlist_file", "playlist_title"):
            if b.get(k):
                evidence[k] = b[k]
        updates.append({"id": sid, "properties": {"public_url": url, "public_url_evidence": evidence}})
    return await journal_curation(
        queue, graph_id, updates=updates, journal_path=journal_path, actor=actor,
        args={"act": "bind-source-urls", "collection_id": collection_id,
              "sources": len(updates)})


async def declare_structure(
    queue: Any,                            # Started job queue
    graph_id: str,                         # Graph-storage capability id
    entries: List[Dict[str, Any]],         # [{"source_id", "structure": {cells}, "evidence": [...]}] — one per member Source
    actor: str,                            # The declaring human (attribution; the map is human-confirmed data)
    journal_path: Optional[str] = None,    # Sidecar journal
    collection_id: Optional[str] = None,   # The Collection the map lies over (semantic summary only)
) -> Dict[str, Any]:  # The journaled op
    """Declare a SOURCE STRUCTURE MAP: the WORK's own part/chapter structure
    laid over a collection's member Sources (DEC 8d9de793 clause 7 with the
    e358a996 erratum). A recording's FILE ORDER and the work's PART/CHAPTER
    structure are two structures over one source and diverge per source (one
    audiobook folds the part readout into a chapter's file, another reads
    section titles as files of their own), so the map is per-source DATA:
    partly derivable from apparatus strata (the 932b78f9 latent-structure
    class), partly human-confirmed, and every cell says which (`evidence`).

    Lands as a `work_structure` property merge on each member Source — the
    PROPERTY form, chosen knowingly as the cheap side of the properties-vs-
    nodes question (a85327b1): the map is an ADDRESSING input (how strata,
    segments and pre-graph notes are lined up for comparison), never the
    deliverable's organizer, and the op journals the intent verbatim so a
    later Part/Chapter node decomposition replays from these same ops.
    Rides `journal_curation` (verb `collection-curation`, act
    `declare-structure`): updates merge idempotently under replay, no new
    handler, and a re-declaration simply overwrites the cells."""
    updates: List[Dict[str, Any]] = []
    for e in entries:
        sid = e["source_id"]
        structure = dict(e.get("structure") or {})
        if not structure:
            raise ValueError(f"declare_structure: entry {sid!r} carries no structure")
        updates.append({"id": sid, "properties": {
            "work_structure": {**structure, "evidence": list(e.get("evidence") or [])}}})
    return await journal_curation(
        queue, graph_id, updates=updates, journal_path=journal_path, actor=actor,
        args={"act": "declare-structure", "collection_id": collection_id,
              "sources": len(updates)})


async def add_reference(
    queue: Any,                            # Started job queue
    graph_id: str,                         # Graph-storage capability id
    source_id: str,                        # The Source (chapter unit) the link enriches — or the work's Collection for a WORK-level link (the work page's Resources; item ebb77107)
    *,
    label: str,                            # Reader-facing link text
    url: str = "",                         # The public URL (also the FALLBACK target while `notes_slug` is not yet born)
    notes_slug: str = "",                  # A notes-graph Note slug this link points at (a cross-work link: resolves to /posts/<slug>/ once born)
    role: str = "related",                 # Open vocabulary; recommended slate: publisher-page · author-post · related-notes · cited-work
    actor: str,                            # The adding human (attribution)
    journal_path: Optional[str] = None,    # Sidecar journal
) -> Dict[str, Any]:  # The journaled op, plus "reference_id"
    """Attach a HUMAN-ADDED RESOURCE LINK to a Source as a `Reference` NODE (ruling
    a7ca900d (3), item ae103970). A link nothing in the audio names — a publisher page,
    the author's post, the notes for a cited work — is an enrichment of the SOURCE, so it
    lives on the transcription graph beside the structure map and every rendering of the
    unit carries it (the compounding rule 1701a235: never authored into a deliverable's
    body). The node id derives from (source, target), so re-adding the same link is a
    no-op under extend-verify; the edge is Source -HAS_REFERENCE-> Reference. A cross-work
    link names a notes-graph slug: the deliverable renderer resolves it to the born page
    when that exists and falls back to `url` until then (finding 962866ae's second
    datapoint). Rides `journal_curation` (act `add-reference`): replays as wires."""
    if not (url or notes_slug):
        raise ValueError("add_reference: give a url and/or a notes_slug")
    if not label.strip():
        raise ValueError("add_reference: a reference needs a reader-facing label")
    target = notes_slug or url
    ref_id = derive_node_id("reference", source_id, target)
    node = {"id": ref_id, "label": TranscriptGraphLabels.REFERENCE, "sources": [],
            "properties": {"source_id": source_id, "label": label.strip(), "url": url,
                           "notes_slug": notes_slug, "role": role, "added_by": actor}}
    edge = make_edge(source_id, ref_id, "HAS_REFERENCE")
    op = await journal_curation(
        queue, graph_id, nodes=[node], edges=[edge], journal_path=journal_path, actor=actor,
        args={"act": "add-reference", "source_id": source_id, "reference_id": ref_id, "role": role})
    op["reference_id"] = ref_id
    return op


async def retract_reference(
    queue: Any,                            # Started job queue
    graph_id: str,                         # Graph-storage capability id
    reference_id: str,                     # The Reference node to retract
    *,
    actor: str,                            # The retracting human (attribution)
    journal_path: Optional[str] = None,    # Sidecar journal
) -> Dict[str, Any]:  # The journaled op
    """Retract a `Reference` node — the compensating act for `add_reference` (the node
    delete cascades its HAS_REFERENCE edge). A link that MOVED is retracted and re-added:
    a new target is a new identity. Rides `journal_curation` (act `retract-reference`);
    an absent node is a tolerated no-op under replay."""
    return await journal_curation(
        queue, graph_id, delete_node_ids=[reference_id], journal_path=journal_path, actor=actor,
        args={"act": "retract-reference", "reference_id": reference_id})
