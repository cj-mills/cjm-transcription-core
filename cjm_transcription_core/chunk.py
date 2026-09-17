"""Chunk-grain re-runs, external landings and the runaway census (work item cf0b91d6; rulings 9ffce5f7 · 8a9b9639 · 910f3692): ONE landing, two producers — a local transcriber re-run of one AudioSegment's rendition, or an operator-pasted external transcript — lands a Transcript variant under the EXISTING AudioSegment with provenance, SUPERSEDES the prior variant for that (rendition, transcriber), journals the delta, and writes a DERIVED run manifest (parent_run_id + per-entry config_hash) the decomp consumes; the census names the chunks in need (degenerate-tail markers on new runs, oversized / implausible words-per-second text on old ones, extreme two-transcriber disagreement) and never counts a superseded variant as live."""

import copy
import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from cjm_context_graph_layer.journal import journal_extend
from cjm_context_graph_layer.ops import graph_task
from cjm_substrate.core.workspace import relativize_recorded, resolve_recorded_tree
from cjm_transcript_graph_schema.schema import (audio_rendition_node_id, audio_segment_node_id,
                                                EXTERNAL_TRANSCRIBER_MARKER, source_node_id,
                                                transcript_node_id, TranscriptNode)

logger = logging.getLogger(__name__)

WORDWRAP_MIN_LINES = 6        # Fewer wrapped-looking lines than this is not a wordwrap shape
WORDWRAP_MIN_RATIO = 0.5      # Share of non-final lines that end mid-sentence for the shape to hold
_SENTENCE_END_CHARS = ".?!:;,\"'”’)]…"

LANDING_VERB = "transcript-landing"  # The journaled op verb for BOTH producers (replays as wires)
PRODUCER_RERUN = "rerun"             # A local transcriber re-run of one chunk (cache bypassed)
PRODUCER_EXTERNAL = "external"       # An operator-pasted transcript from an external model


def load_run_manifest(
    path: Any,  # Path to a transcription-core run manifest JSON
) -> Dict[str, Any]:  # Parsed manifest dict, ${WS}/ paths resolved to absolute
    """Load a transcription-core run manifest (${WS}/ recorded paths resolve at load,
    anchored at the manifest's own location — the 5daadfc4 reader half). Refuses a
    foreign format loudly: the derived-manifest chain must start from OUR manifest."""
    p = Path(path)
    data = resolve_recorded_tree(json.loads(p.read_text()), p)
    fmt = str(data.get("format") or "")
    if "transcription-core" not in fmt:
        raise ValueError(f"{p}: not a transcription-core run manifest (format {fmt!r})")
    if not isinstance(data.get("sources"), list):
        raise ValueError(f"{p}: manifest has no sources list")
    return data


def select_chunks(
    manifest: Dict[str, Any],                # A loaded run manifest
    source: Optional[str] = None,            # None = every source; else an index, a content-hash prefix, or a source_path substring
    segments: Optional[List[int]] = None,    # None = every segment of the selected sources; else segment indices
) -> List[Tuple[int, Dict[str, Any], Dict[str, Any]]]:  # (source_index, source_entry, segment_entry) rows
    """Resolve a chunk selection against the manifest (pure).

    A `source` that is all digits selects by index; otherwise it matches a
    content-hash prefix (`sha256:...` or the bare hex) or a substring of the
    source path — and MUST resolve to exactly one source when it is given."""
    srcs = list(manifest.get("sources") or [])
    picked: List[Tuple[int, Dict[str, Any]]]
    if source is None:
        picked = list(enumerate(srcs))
    elif source.strip().isdigit():
        i = int(source)
        if not 0 <= i < len(srcs):
            raise ValueError(f"source index {i} out of range (manifest has {len(srcs)} sources)")
        picked = [(i, srcs[i])]
    else:
        key = source.strip()
        bare = key.split(":", 1)[1] if key.startswith("sha256:") else key
        picked = [(i, s) for i, s in enumerate(srcs)
                  if str(s.get("content_hash") or "").startswith("sha256:" + bare)
                  or key in str(s.get("source_path") or "")]
        if len(picked) != 1:
            raise ValueError(f"source {source!r} matches {len(picked)} sources (need exactly one): "
                             f"{[str(s.get('source_path')) for _, s in picked][:5]}")
    want = set(int(x) for x in (segments or []))
    rows: List[Tuple[int, Dict[str, Any], Dict[str, Any]]] = []
    for si, s in picked:
        for seg in s.get("segments") or []:
            if not want or int(seg.get("index", -1)) in want:
                rows.append((si, s, seg))
    if want and len(rows) < len(want) * len(picked):
        found = {int(seg.get("index", -1)) for _, _, seg in rows}
        raise ValueError(f"segments {sorted(want - found)} not found in the selected source(s)")
    return rows


def prior_config_hash(
    manifest: Dict[str, Any],      # The (parent) run manifest
    seg: Dict[str, Any],           # The segment entry
    transcriber: str,              # Transcriber name (manifest `transcripts` key)
) -> str:  # The config hash the chunk's CURRENT variant carries ("" = none recorded / no variant)
    """The config hash of the variant this transcriber currently has on the chunk: a
    per-entry override (a derived manifest's earlier landing) wins over the run-level
    capabilities block; "" when the transcriber has no text on the chunk."""
    entry = (seg.get("transcripts") or {}).get(transcriber)
    if entry is None:
        return ""
    if entry.get("config_hash"):
        return str(entry["config_hash"])
    return str(((manifest.get("capabilities") or {}).get(transcriber) or {}).get("config_hash") or "")


def text_shape(
    text: str,  # A pasted transcript, verbatim
) -> Dict[str, Any]:  # {lines, paragraphs, wrapped, wordwrap: bool} — the landing's newline census
    """Census the newline shape of a pasted external transcript (finding efe88f17;
    pure). A `paragraph` is a blank-line break the model wrote on purpose; a
    `wrapped` line is a non-final line of a paragraph that ends without sentence
    punctuation — the fixed-width word-wrap the AI Studio 'Copy as text' gesture
    bakes in ('Copy as markdown' carries none). `wordwrap` holds when at least
    WORDWRAP_MIN_LINES such lines exist and they are at least WORDWRAP_MIN_RATIO
    of the non-final lines. The text is never changed here — the landing stays
    verbatim; decomp's fold normalises its reading of an external variant."""
    paragraphs = [p for p in re.split(r"\n[ \t]*\n", text) if p.strip()]
    lines = 0
    non_final = 0
    wrapped = 0
    for p in paragraphs:
        plines = [ln.rstrip() for ln in p.split("\n") if ln.strip()]
        lines += len(plines)
        for ln in plines[:-1]:
            non_final += 1
            if ln[-1] not in _SENTENCE_END_CHARS:
                wrapped += 1
    ratio = (wrapped / non_final) if non_final else 0.0
    return {"lines": lines, "paragraphs": len(paragraphs), "wrapped": wrapped,
            "wordwrap": wrapped >= WORDWRAP_MIN_LINES and ratio >= WORDWRAP_MIN_RATIO}


def wordwrap_warning(
    shape: Dict[str, Any],  # A `text_shape` census
) -> Optional[str]:  # The operator-facing warning, or None when the shape is clean
    """The landing-time warning for a wordwrap-shaped paste (efe88f17 (2)): the
    text lands verbatim either way; the operator learns the craft."""
    if not shape.get("wordwrap"):
        return None
    return (f"WARNING: wordwrap shape — {shape['wrapped']} of {shape['lines']} lines end mid-sentence "
            f"(fixed-width wraps from a 'Copy as text' paste). The text lands verbatim and decomp "
            f"folds it with wraps read as spaces; next time use 'Copy as markdown' in AI Studio, "
            f"which carries no wrap newlines.")


def build_chunk_landing(
    source_entry: Dict[str, Any],     # The manifest source entry (content_hash, chain)
    seg: Dict[str, Any],              # The manifest segment entry (start, end, model_input_hash)
    *,
    transcriber: str,                 # Transcriber name the variant files under
    config_hash: str,                 # The NEW variant's config hash (identity input)
    text: str,                        # The transcript text
    metadata: Optional[Dict[str, Any]] = None,  # Transcriber-reported metadata + the landing provenance
    prior_config_hash: str = "",      # The prior variant's config hash ("" or equal = nothing to supersede)
    actor: Optional[str] = None,      # Attribution actor (None = capability:<transcriber>)
    method: str = "transcribe",       # Attribution method
    asserted_at: Optional[float] = None,  # Derivation timestamp; None = now
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:  # (nodes, edges, ids)
    """Build the landing payload for ONE chunk (pure; no capability calls).

    Recomputes the root ids the way emission and decomp do (Source = content hash;
    AudioSegment = (source, range); AudioRendition = (segment, chain)), mints the
    Transcript variant under the EXISTING rendition (DERIVED_FROM), and — when a
    prior variant with a different config hash exists for the same (rendition,
    transcriber) — the SUPERSEDES edge new -> prior (ruling 910f3692 (2): the old
    node stays; readers filter on the incoming edge). Idempotent under
    extend-verify: the same landing twice collides into a verified no-op."""
    content_hash = str(source_entry.get("content_hash") or "")
    if not content_hash:
        raise ValueError(f"source {source_entry.get('source_path')!r} has no content_hash — landing identity requires it")
    model_input_hash = str(seg.get("model_input_hash") or "")
    if not model_input_hash:
        raise ValueError(f"segment {seg.get('index')} has no model_input_hash — landing identity requires it")
    if not config_hash:
        raise ValueError("build_chunk_landing: config_hash is required (the variant's identity input)")
    chain = list(source_entry.get("chain") or [])
    source_id = source_node_id(content_hash)
    aseg_id = audio_segment_node_id(source_id, float(seg.get("start", 0.0)), float(seg.get("end", 0.0)))
    rendition_id = audio_rendition_node_id(aseg_id, chain)
    tnode = TranscriptNode(rendition=rendition_id, transcriber=transcriber, config_hash=config_hash,
                           text=text, audio_hash=model_input_hash, metadata=dict(metadata or {}),
                           asserted_at=asserted_at, actor=actor, method=method)
    nodes = [tnode.to_graph_node()]
    edges = [tnode.derived_edge()]
    prior_id: Optional[str] = None
    if prior_config_hash and prior_config_hash != config_hash:
        prior_id = transcript_node_id(rendition_id, transcriber, prior_config_hash)
        edges.append(tnode.supersedes_edge(prior_id))
    ids = {"source": source_id, "audio_segment": aseg_id, "rendition": rendition_id,
           "transcript": tnode.id, "supersedes": prior_id}
    return nodes, edges, ids


async def land_chunk_transcript(
    queue: Any,                       # Started job queue
    graph_id: str,                    # Graph-storage capability instance id
    source_entry: Dict[str, Any],     # The manifest source entry
    seg: Dict[str, Any],              # The manifest segment entry
    *,
    transcriber: str,                 # Transcriber name the variant files under
    config_hash: str,                 # The new variant's config hash
    text: str,                        # The transcript text
    metadata: Optional[Dict[str, Any]] = None,  # Metadata incl. the `landing` provenance block
    prior_config_hash: str = "",      # The prior variant's config hash ("" = none)
    producer: str,                    # PRODUCER_RERUN | PRODUCER_EXTERNAL
    reason: str,                      # Why (runaway / operator / escalation ...)
    actor: str,                       # Who initiated (journal attribution)
    run_id: str,                      # The landing run's id (pins the derived manifest)
    journal_path: Optional[str] = None,  # Sidecar write journal (None = unjournaled)
    node_actor: Optional[str] = None, # Transcript attribution actor (None = capability:<transcriber>)
    method: str = "transcribe",       # Transcript attribution method
) -> Dict[str, Any]:  # {"transcript", "supersedes", "audio_segment", "rendition", "source", nodes/edges counts}
    """Land one chunk's variant through the task channel and journal the delta —
    the ONE landing both producers share (the op verb is `transcript-landing`,
    replayed as wires by the transcription core's handler vocabulary)."""
    nodes, edges, ids = build_chunk_landing(
        source_entry, seg, transcriber=transcriber, config_hash=config_hash, text=text,
        metadata=metadata, prior_config_hash=prior_config_hash, actor=node_actor, method=method)
    res = await journal_extend(queue, graph_id, nodes, edges, journal_path=journal_path,
                               verb=LANDING_VERB, actor=actor, run=run_id,
                               args={"producer": producer, "reason": reason,
                                     "source_id": ids["source"], "audio_segment": ids["audio_segment"],
                                     "segment_index": int(seg.get("index", -1)),
                                     "transcriber": transcriber, "transcript_id": ids["transcript"],
                                     "supersedes": ids["supersedes"]})
    return {**ids, "nodes_added": res.nodes_added, "nodes_verified": res.nodes_verified,
            "edges_added": res.edges_added, "edges_existing": res.edges_existing}


def derive_manifest(
    parent: Dict[str, Any],   # The loaded parent manifest (paths resolved)
    *,
    run_id: str,              # The derived manifest's own run id
    parent_path: Any,         # Where the parent was loaded from (recorded for the chain)
    kind: str,                # "rerun-chunk" | "add-transcript"
    created_at: Optional[float] = None,  # None = now
) -> Dict[str, Any]:  # A deep copy of the parent with the derivation header set
    """Start a DERIVED manifest (ruling 910f3692 (1)): a copy of the parent under a NEW
    run id, `parent_run_id` + `parent_manifest` naming what it derives from, every
    entry untouched until `apply_chunk_update` replaces the ones this run landed —
    the parent stays byte-identical, the derived file is what decomp consumes."""
    d = copy.deepcopy(parent)
    d["run_id"] = run_id
    d["created_at"] = float(created_at if created_at is not None else time.time())
    d["parent_run_id"] = str(parent.get("run_id") or "")
    d["parent_manifest"] = str(parent_path)
    d["derivation"] = {"kind": kind, "landings": []}
    return d


def apply_chunk_update(
    derived: Dict[str, Any],          # The derived manifest (mutated in place)
    source_index: int,                # Source position in the manifest
    segment_index: int,               # Segment `index` within that source
    transcriber: str,                 # Transcriber name (manifest `transcripts` key)
    entry: Dict[str, Any],            # The new transcripts entry: {job_id, text, metadata, config_hash, landing}
    *,
    capability_info: Optional[Dict[str, Any]] = None,  # Run-level capabilities record for a transcriber NEW to the run
) -> Dict[str, Any]:  # The landing row appended to derivation.landings
    """Replace ONE chunk's entry for ONE transcriber in the derived manifest.

    The entry carries its OWN `config_hash` (decomp's id recomputation honours the
    per-entry override over the run-level block), so a re-run under a new config on
    a few chunks leaves every other chunk's identity untouched. A transcriber NEW to
    the run (an external landing) is registered in the run's transcriber list and,
    when `capability_info` is given, in the capabilities block."""
    if not entry.get("config_hash"):
        raise ValueError("apply_chunk_update: the entry needs its own config_hash")
    src = derived["sources"][source_index]
    seg = next((s for s in src.get("segments") or [] if int(s.get("index", -1)) == segment_index), None)
    if seg is None:
        raise ValueError(f"segment {segment_index} not in source {source_index}")
    seg.setdefault("transcripts", {})[transcriber] = dict(entry)
    cfg = derived.setdefault("config", {})
    names = list(cfg.get("transcriber_capabilities") or [])
    if transcriber not in names:
        names.append(transcriber)
        cfg["transcriber_capabilities"] = names
    caps = derived.setdefault("capabilities", {})
    if transcriber not in caps and capability_info is not None:
        caps[transcriber] = dict(capability_info)
    row = {"source_index": source_index, "segment_index": segment_index, "transcriber": transcriber,
           "config_hash": entry["config_hash"], **dict(entry.get("landing") or {})}
    derived.setdefault("derivation", {}).setdefault("landings", []).append(row)
    return row


def save_manifest(
    derived: Dict[str, Any],  # The derived manifest
    path: Any,                # Destination JSON file (parent dirs created)
    workspace: Any = None,    # Active Workspace; owned paths record as ${WS}/<rel>
) -> Path:  # The written path
    """Write the derived manifest (the same recording contract as RunManifest.save)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(relativize_recorded(derived, workspace), indent=2))
    return out


def prompt_hash_of(
    text: str,  # The prompt template text
) -> str:  # "sha256:<hex>" over the normalized prompt (trailing whitespace stripped)
    """Hash a prompt template — the prompt is DATA (f304d31d) and its hash rides the
    external landing's config hash, so a different prompt is a different variant."""
    return "sha256:" + hashlib.sha256(text.rstrip().encode("utf-8")).hexdigest()


# ---- the runaway census -------------------------------------------------------------

TRANSCRIPT_ROWS_SQL = """
WITH tr AS (
  SELECT n.id AS transcript_id,
         json_extract(n.properties, '$.transcriber') AS transcriber,
         json_extract(n.properties, '$.config_hash') AS config_hash,
         json_extract(n.properties, '$.rendition_id') AS rendition_id,
         length(coalesce(json_extract(n.properties, '$.text'), '')) AS chars,
         CASE WHEN coalesce(json_extract(n.properties, '$.text'), '') = '' THEN 0
              ELSE length(json_extract(n.properties, '$.text'))
                   - length(replace(json_extract(n.properties, '$.text'), ' ', '')) + 1 END AS words,
         (json_extract(n.properties, '$.metadata.degenerate_tail') IS NOT NULL) AS degenerate,
         (instr(coalesce(json_extract(n.properties, '$.text'), ''), 'ms ]') > 0) AS timestamp_leak,
         json_extract(n.properties, '$.metadata.landing.producer') AS producer,
         EXISTS (SELECT 1 FROM edges s WHERE s.target_id = n.id AND s.relation_type = 'SUPERSEDES') AS superseded
  FROM nodes n WHERE n.label = 'Transcript'
), rend AS (
  SELECT e.source_id AS rendition_id, e.target_id AS aseg_id
  FROM edges e JOIN nodes a ON a.id = e.target_id
  WHERE e.relation_type = 'DERIVED_FROM' AND a.label = 'AudioSegment'
), aseg AS (
  SELECT a.id AS aseg_id,
         json_extract(a.properties, '$.index') AS seg_index,
         json_extract(a.properties, '$.start') AS start,
         json_extract(a.properties, '$.end') AS "end",
         (SELECT p.target_id FROM edges p JOIN nodes s ON s.id = p.target_id
            WHERE p.source_id = a.id AND p.relation_type = 'PART_OF' AND s.label = 'Source' LIMIT 1) AS source_id
  FROM nodes a WHERE a.label = 'AudioSegment'
), src AS (
  SELECT s.id AS source_id,
         json_extract(s.properties, '$.path') AS source_path,
         json_extract(s.properties, '$.content_hash') AS content_hash,
         (SELECT json_extract(c.properties, '$.title') FROM edges p JOIN nodes c ON c.id = p.target_id
            WHERE p.source_id = s.id AND p.relation_type = 'PART_OF' AND c.label = 'Collection' LIMIT 1) AS collection,
         (SELECT json_extract(c.properties, '$.status') FROM edges p JOIN nodes c ON c.id = p.target_id
            WHERE p.source_id = s.id AND p.relation_type = 'PART_OF' AND c.label = 'Collection' LIMIT 1) AS collection_status
  FROM nodes s WHERE s.label = 'Source'
)
SELECT src.collection, src.collection_status, src.source_id, src.source_path, src.content_hash,
       aseg.aseg_id AS audio_segment, aseg.seg_index, aseg.start, aseg."end",
       tr.rendition_id, tr.transcriber, tr.config_hash, tr.transcript_id,
       tr.chars, tr.words, tr.degenerate, tr.producer, tr.superseded
FROM tr JOIN rend ON rend.rendition_id = tr.rendition_id
        JOIN aseg ON aseg.aseg_id = rend.aseg_id
        JOIN src ON src.source_id = aseg.source_id
ORDER BY src.collection, src.source_path, aseg.seg_index, tr.transcriber
"""


async def fetch_transcript_rows(
    queue: Any,       # Started job queue
    graph_id: str,    # Graph-storage capability instance id
) -> List[Dict[str, Any]]:  # One row per Transcript node with its chunk / source / collection context
    """Pull every Transcript's census inputs through the graph capability's marked
    raw read (`raw_query`, backend sqlite — the SG-41 read-only escape; the typed
    query surface cannot yet express the four-hop join) — sizes and markers only,
    never the text bodies."""
    res = await graph_task(queue, graph_id, "raw_query",
                           query={"type": "raw_query", "text": TRANSCRIPT_ROWS_SQL,
                                  "backend": "sqlite", "params": []})
    cols = list(res.get("columns") or []) if isinstance(res, dict) else list(getattr(res, "columns", []))
    rows = list(res.get("rows") or []) if isinstance(res, dict) else list(getattr(res, "rows", []))
    return [dict(zip(cols, r)) for r in rows]


def census_rows(
    rows: List[Dict[str, Any]],            # Transcript rows (fetch_transcript_rows shape)
    *,
    max_chars: int = 20000,                # Oversized-text threshold (the finding's > 20000-char census)
    max_words_per_second: float = 8.0,     # Implausible speech rate over the chunk's duration (decomp's fold gate)
    disagreement_ratio: float = 4.0,       # Live word-count ratio between two transcribers on one rendition that flags both
    min_words: int = 20,                   # Disagreement needs at least this many words on the longer side
    transcriber: Optional[str] = None,     # Restrict the flagged rows to one transcriber (context rows still count)
    collections: Optional[List[str]] = None,  # Restrict to these collection titles (None = all)
    include_superseded: bool = False,      # True = superseded variants are listed too (marked); default: not live
    include_escalated: bool = False,       # True = chunks already carrying an external (/manual) variant are listed too (marked `escalated`)
    include_retired: bool = False,         # True = sources of RETIRED collections (ruling a7617bd4) count too; default: hidden
) -> List[Dict[str, Any]]:  # Flagged rows: each carries `reasons` (list) + `words_per_second` + `escalated`
    """The runaway census (pure): which LIVE chunk variants need a better transcription.

    Reasons: `degenerate` (the 0.0.49 guard's marker on a new run), `oversized`
    (> max_chars — the old-run signature), `implausible_rate` (words/s over the
    chunk's duration > max_words_per_second), `disagreement` (two live transcribers
    on one rendition differ by more than `disagreement_ratio` in word count — the
    SHORTER side is the suspect on a runaway, so both rows are listed with the
    ratio). A superseded variant is NOT live: excluded unless asked for. A chunk
    that already carries a LIVE external variant (`<model id>/manual`) is
    ESCALATED — covered by the operator — and drops out unless asked for; the
    external variant itself is never flagged (it is the operator's answer)."""
    wanted = set(c for c in (collections or []))
    live = [r for r in rows if include_superseded or not r.get("superseded")]
    if not include_retired:  # a retired collection is hidden, never cascaded (a7617bd4)
        live = [r for r in live if str(r.get("collection_status") or "") != "retired"]
    if wanted:
        live = [r for r in live if (r.get("collection") or "") in wanted]
    by_rendition: Dict[str, List[Dict[str, Any]]] = {}
    for r in live:
        by_rendition.setdefault(str(r.get("rendition_id")), []).append(r)
    flagged: List[Dict[str, Any]] = []
    for r in live:
        if is_external_transcriber(str(r.get("transcriber") or "")):
            continue
        escalated = any(is_external_transcriber(str(p.get("transcriber") or "")) and not p.get("superseded")
                        for p in by_rendition.get(str(r.get("rendition_id")), []))
        if escalated and not include_escalated:
            continue
        dur = float(r.get("end") or 0.0) - float(r.get("start") or 0.0)
        words = float(r.get("words") or 0)
        wps = (words / dur) if dur > 0 else 0.0
        reasons: List[str] = []
        if r.get("degenerate"):
            reasons.append("degenerate")
        if r.get("timestamp_leak"):
            reasons.append("timestamp_leak")  # the model emitted '[ 0m2s22ms ]' spans instead of prose (2026-09-12 field sighting)
        if int(r.get("chars") or 0) > max_chars:
            reasons.append("oversized")
        if wps > max_words_per_second:
            reasons.append("implausible_rate")
        ratio = None
        peers = [p for p in by_rendition.get(str(r.get("rendition_id")), []) if p is not r]
        for p in peers:
            a, b = words, float(p.get("words") or 0)
            hi, lo = max(a, b), min(a, b)
            if hi >= min_words and (lo == 0 or hi / lo > disagreement_ratio):
                ratio = (hi / lo) if lo else float("inf")
                reasons.append("disagreement")
                break
        if not reasons:
            continue
        if transcriber and r.get("transcriber") != transcriber:
            continue
        flagged.append({**r, "duration": dur, "words_per_second": round(wps, 2),
                        "disagreement_ratio": (None if ratio is None else (round(ratio, 1) if ratio != float("inf") else "inf")),
                        "reasons": reasons, "escalated": escalated})
    return flagged


def is_external_transcriber(
    name: str,  # A transcriber name (manifest `transcripts` key / Transcript.transcriber)
) -> bool:  # True for an operator-landed external variant (`<model id>/manual`)
    """Whether a transcriber name is an external landing's (the `/manual` marker)."""
    return name.endswith(EXTERNAL_TRANSCRIBER_MARKER)


def summarize_census(
    flagged: List[Dict[str, Any]],  # census_rows output
    rows: Optional[List[Dict[str, Any]]] = None,  # All transcript rows (for the live totals per collection)
) -> Dict[str, Dict[str, Any]]:  # collection -> {flagged, live, by_reason, by_transcriber}
    """Per-collection roll-up of the census (the closing evidence for 56a802b3 is a
    zero `flagged` on every LIVE collection)."""
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows or []:
        # Every collection seen is listed (a fully superseded one shows live 0).
        c = out.setdefault(str(r.get("collection") or ""), {"flagged": 0, "live": 0, "by_reason": {}, "by_transcriber": {}})
        if not r.get("superseded"):
            c["live"] += 1
    for f in flagged:
        c = out.setdefault(str(f.get("collection") or ""), {"flagged": 0, "live": 0, "by_reason": {}, "by_transcriber": {}})
        c["flagged"] += 1
        for reason in f.get("reasons") or []:
            c["by_reason"][reason] = c["by_reason"].get(reason, 0) + 1
        t = str(f.get("transcriber") or "")
        c["by_transcriber"][t] = c["by_transcriber"].get(t, 0) + 1
    return out


def rows_from_manifest(
    manifest: Dict[str, Any],  # A loaded run manifest
) -> List[Dict[str, Any]]:  # Census rows in the fetch_transcript_rows shape, straight from the manifest
    """Census inputs from a run manifest alone (no graph): the qt inspection lane's
    offline source. The rendition key is (source index, segment index); a derived
    manifest's per-entry config hash and landing provenance ride along; nothing
    here is superseded (a manifest holds one variant per transcriber per chunk)."""
    coll = ", ".join(str(c.get("title") or "") for c in (manifest.get("collections") or []) if c.get("title"))
    caps = manifest.get("capabilities") or {}
    rows: List[Dict[str, Any]] = []
    for si, s in enumerate(manifest.get("sources") or []):
        content_hash = str(s.get("content_hash") or "")
        sid = source_node_id(content_hash) if content_hash else f"src:{si}"
        for seg in s.get("segments") or []:
            idx = int(seg.get("index", -1))
            for t, entry in (seg.get("transcripts") or {}).items():
                text = str((entry or {}).get("text") or "")
                meta = dict((entry or {}).get("metadata") or {})
                rows.append({
                    "collection": coll, "source_id": sid, "source_index": si,
                    "source_path": str(s.get("source_path") or ""), "content_hash": content_hash,
                    "audio_segment": None, "seg_index": idx,
                    "start": float(seg.get("start", 0.0)), "end": float(seg.get("end", 0.0)),
                    "rendition_id": f"{si}:{idx}", "transcriber": t,
                    "config_hash": str((entry or {}).get("config_hash") or (caps.get(t) or {}).get("config_hash") or ""),
                    "transcript_id": None, "chars": len(text), "words": len(text.split()),
                    "degenerate": 1 if meta.get("degenerate_tail") else 0,
                    "producer": (meta.get("landing") or {}).get("producer"), "superseded": 0,
                })
    return rows


def flagged_chunks(
    manifest: Dict[str, Any],  # A loaded run manifest
    **thresholds: Any,         # census_rows keyword thresholds (max_chars, max_words_per_second, include_escalated, ...)
) -> Dict[Tuple[int, int], List[Dict[str, Any]]]:  # (source_index, seg_index) -> that chunk's flagged rows, in manifest order
    """The inspection lane's jump index: which chunks of a run carry a flagged variant,
    keyed by (source index, segment index) in manifest order (ruling 8a9b9639 (1):
    total failures first — the census's reasons — but ANY chunk stays escalatable).
    A chunk the operator already escalated (an external variant beside the flagged
    one) is out of the index unless `include_escalated=True` — then its rows carry
    `escalated: True` so the lane can show it as covered."""
    out: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for f in census_rows(rows_from_manifest(manifest), **thresholds):
        out.setdefault((int(f["source_index"]), int(f["seg_index"])), []).append(f)
    return dict(sorted(out.items()))


DEFAULT_ESCALATION_MODEL_ID = "gemini-3.8-flash"  # The external model id every escalation import prefills (user ruling ad0e3b4f, 2026-09-16): ONE constant, both shells + the CLI help read it

DEFAULT_ESCALATION_PROMPT = """You are transcribing one chunk of a recording. Produce a verbatim, punctuated transcript of the attached audio — nothing else: no summary, no speaker labels, no timestamps, no commentary.

Context (use it to resolve names, jargon and acronyms you hear):
- Recording: {source_title}
- Collection: {collection}
- Chunk: {chunk_range} of the recording
- Text just BEFORE this chunk (another transcriber): {prev_text}
- Text just AFTER this chunk (another transcriber): {next_text}
- Current draft of THIS chunk (may be wrong or truncated): {draft_text}
- Glossary / known terms: {glossary}

Rules: keep filler words, keep false starts if audible; if a passage is inaudible write [inaudible]; do not invent content; do not repeat a phrase more than it is spoken.
"""


def render_escalation_prompt(
    manifest: Dict[str, Any],      # A loaded run manifest
    source_index: int,             # Source position in the manifest
    seg_index: int,                # Segment `index` within that source
    *,
    template: Optional[str] = None,      # The prompt TEMPLATE (None = DEFAULT_ESCALATION_PROMPT); its hash is the variant's prompt hash
    transcriber: Optional[str] = None,   # Whose text fills the neighbour / draft slots when a chunk carries no external landing (None = the first transcriber with text)
    glossary: Optional[List[str]] = None,  # Known terms to carry (f9d0fd93 — the escalation output doubles as glossary evidence)
    neighbour_chars: int = 600,          # Tail / head of the neighbouring chunks' text to include
    draft_chars: int = 1200,             # Head of the chunk's own current text to include
    slot_text: Optional[Callable[[float, float], Optional[str]]] = None,  # LIVE slot-text provider (chunk start, end) -> the effective text over that span, or None/"" = fall back to the manifest (finding c63cd2e3)
) -> Dict[str, Any]:  # {"prompt", "prompt_hash", "template", "slots", "slot_sources"}
    """Render the escalation prompt WITH CONTEXT for one chunk (ruling 8a9b9639 (3):
    'nickel' is NCCL in 'Lecture 17: NCCL' — context is what audio cannot give). The
    prompt is DATA (f304d31d): the TEMPLATE's hash — not the rendered text's — rides
    the external landing's config hash, so one template = one variant identity across
    chunks. The rendered prompt is what the copy-paste gesture puts on the clipboard.

    Slot text follows PER-CHUNK AUTHORITY (ruling cad12c97): a chunk that already
    carries an external landing (`<model id>/manual`) lends THAT text to its
    neighbours' context and to its own draft — the escalated transcript is the
    chunk's text source, so the next chunk's prompt reads the better text (user
    sighting 2026-09-16: the second chunk's prompt quoted whisper's tail of the
    first, not the imported one). Otherwise `transcriber` (the caller's accuracy
    model), else the first transcriber with text in manifest order.

    A LIVE provider outranks all of that (finding c63cd2e3, user ruling
    2026-09-17): when the caller holds the corrected spine (the correction app's
    E gesture), `slot_text(start, end)` fills the BEFORE / AFTER / DRAFT slots
    with the effective text over each chunk's span — manual fidelity edits and
    landed escalations included — and the manifest answers only where the
    provider returns nothing. `slot_sources` names each slot's origin
    ("spine" | "manifest"); the template hash is untouched either way (slot
    text never rides the variant identity)."""
    tpl = template if template is not None else DEFAULT_ESCALATION_PROMPT
    srcs = list(manifest.get("sources") or [])
    if not 0 <= source_index < len(srcs):
        raise ValueError(f"source index {source_index} out of range")
    src = srcs[source_index]
    segs = list(src.get("segments") or [])
    pos = next((i for i, s in enumerate(segs) if int(s.get("index", -1)) == seg_index), None)
    if pos is None:
        raise ValueError(f"segment {seg_index} not in source {source_index}")
    seg = segs[pos]

    def text_of(s: Optional[Dict[str, Any]]) -> str:
        if s is None:
            return ""
        tr = s.get("transcripts") or {}
        order = list((manifest.get("config") or {}).get("transcriber_capabilities") or list(tr))
        # Per-chunk authority first: the LAST external landing with text on this chunk.
        for t in reversed(order):
            if is_external_transcriber(t):
                txt = str((tr.get(t) or {}).get("text") or "")
                if txt:
                    return txt
        if transcriber and transcriber in tr:
            return str((tr[transcriber] or {}).get("text") or "")
        for t in order:
            txt = str((tr.get(t) or {}).get("text") or "")
            if txt:
                return txt
        return ""

    slot_sources: Dict[str, str] = {}

    def slot_of(name: str, s: Optional[Dict[str, Any]]) -> str:
        """Live spine text over the chunk's span first, the manifest's text second."""
        if s is not None and slot_text is not None:
            live = slot_text(float(s.get("start", 0.0)), float(s.get("end", 0.0)))
            if live and str(live).strip():
                slot_sources[name] = "spine"
                return str(live)
        slot_sources[name] = "manifest" if s is not None else "none"
        return text_of(s)

    prev_text = slot_of("prev_text", segs[pos - 1] if pos > 0 else None)[-neighbour_chars:].strip()
    next_text = slot_of("next_text", segs[pos + 1] if pos + 1 < len(segs) else None)[:neighbour_chars].strip()
    draft = slot_of("draft_text", seg)[:draft_chars].strip()
    slots = {
        "source_title": Path(str(src.get("source_path") or "")).stem or "(untitled)",
        "collection": ", ".join(str(c.get("title") or "") for c in (manifest.get("collections") or [])) or "(none)",
        "chunk_range": f"{float(seg.get('start', 0.0)):.0f}s-{float(seg.get('end', 0.0)):.0f}s",
        "prev_text": prev_text or "(none)",
        "next_text": next_text or "(none)",
        "draft_text": draft or "(none)",
        "glossary": ", ".join(glossary) if glossary else "(none)",
    }
    prompt = tpl.format(**slots)
    return {"prompt": prompt, "prompt_hash": prompt_hash_of(tpl), "template": tpl, "slots": slots,
            "slot_sources": slot_sources}


def chunks_from_census(
    manifest: Dict[str, Any],             # The loaded run manifest
    flagged: List[Dict[str, Any]],        # census_rows output
    transcriber: str,                     # The transcriber whose flagged variants select the chunks
) -> List[Tuple[int, Dict[str, Any], Dict[str, Any]]]:  # (source_index, source_entry, segment_entry) rows, manifest order
    """Map census rows onto the manifest's chunks by (Source id, segment index) — the
    `--flagged` target list of a re-run (pure)."""
    keys = {(str(f.get("source_id")), int(f.get("seg_index") if f.get("seg_index") is not None else -1))
            for f in flagged if f.get("transcriber") == transcriber}
    rows: List[Tuple[int, Dict[str, Any], Dict[str, Any]]] = []
    for si, s in enumerate(manifest.get("sources") or []):
        sid = source_node_id(str(s.get("content_hash") or "")) if s.get("content_hash") else None
        if sid is None:
            continue
        for seg in s.get("segments") or []:
            if (sid, int(seg.get("index", -1))) in keys:
                rows.append((si, s, seg))
    return rows
