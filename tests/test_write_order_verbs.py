"""THE PER-VERB CRASH PROOF of the transcription core (eeda5977 leg L3 slice E3, rulings 6683f49e +
3de795e4), on the layer's harness (`cjm_context_graph_layer.testing.run_proof`).

Every journaled transcription verb runs as a step in a child process on a real SQLite graph,
uncrashed and then crashed at every op boundary. The pipeline's writes -- a source emission and the
Derivation it plans, the run's collection declarations, a chunk's transcript landing -- are RE-RUN
after a crash, their plans finding what the crash left (547f4123); a re-emission journals nothing
(17daa86b). The hub's curation verbs are human acts, healed by the next writer, and the hub's
two-group file gesture is ONE unit (ruling 3de795e4). Each run's live db equals its journal's
replay element for element, and every crashed run's journal equals the uncrashed one, op for op.

Needs the graph capability: CJM_PROOF_MANIFESTS, else this repo's .cjm/manifests."""

import os
from pathlib import Path

import pytest
from cjm_context_graph_layer.journal import write_window
from cjm_context_graph_layer.testing import crash_variants, ProofStep, run_proof
from cjm_transcript_graph_schema.schema import CollectionNode, source_node_id
from cjm_transcription_core import curation as C
from cjm_transcription_core.chunk import land_chunk_transcript
from cjm_transcription_core.emission import emit_collections_graph, emit_source_graph
from cjm_transcription_core.models import CollectionDecl, SegmentRecord, SourceResult

MANIFESTS = os.environ.get("CJM_PROOF_MANIFESTS") or str(Path(__file__).resolve().parents[1] / ".cjm" / "manifests")
pytestmark = pytest.mark.skipif(not (Path(MANIFESTS) / "cjm-capability-graph-sqlite.json").exists(),
                                reason=f"graph capability not installed at {MANIFESTS}")

ACTOR = "human:proof"
MEDIA = ["/media/proof-a.mp3", "/media/proof-b.mp3"]
HASHES = ["sha256:proof-src-a", "sha256:proof-src-b"]
SRC = [source_node_id(h) for h in HASHES]
DECLARED, SERIES = "Proof Declared", "Proof Series"


def _j(ctx):
    return {"journal_path": ctx.journal}


def _coll(title):  # A collection's id is its title's (create-or-attach by title)
    return CollectionNode(title=title).id


def _source(k):  # One source, two chunks, one transcriber
    segs = [SegmentRecord(index=i, start=280.0 * i, end=280.0 * (i + 1), duration=280.0,
                          segment_path=f"/cuts/p{k}{i}.mp3", model_input_path=f"/cache/p{k}{i}.wav",
                          model_input_hash=f"sha256:proof-wav{k}{i}",
                          transcripts={"whisper": {"job_id": f"j{k}{i}", "text": f"chunk {i}", "metadata": {}}})
            for i in range(2)]
    return SourceResult(source_path=MEDIA[k], duration=560.0, vad_chunk_count=2, batch_key="bk",
                        content_hash=HASHES[k], segments=segs)


async def emit_a(ctx):
    await emit_source_graph(ctx.queue, ctx.graph_id, _source(0), {"whisper": "sha256:proof-cfg"},
                            "run-proof", **_j(ctx))


async def emit_b(ctx):
    await emit_source_graph(ctx.queue, ctx.graph_id, _source(1), {"whisper": "sha256:proof-cfg"},
                            "run-proof", **_j(ctx))


async def declare(ctx):
    await emit_collections_graph(ctx.queue, ctx.graph_id,
                                 [CollectionDecl(title=DECLARED, member_paths=list(MEDIA), ordered=True)],
                                 dict(zip(MEDIA, SRC)), "run-proof", **_j(ctx))


async def land(ctx):
    await land_chunk_transcript(
        ctx.queue, ctx.graph_id, {"source_path": MEDIA[0], "content_hash": HASHES[0], "chain": []},
        {"index": 1, "start": 280.0, "end": 560.0, "model_input_path": "/cache/p01.wav",
         "model_input_hash": "sha256:proof-wav01"},
        transcriber="whisper", config_hash="sha256:proof-cfg-new", text="chunk one, rerun",
        prior_config_hash="sha256:proof-cfg", producer="rerun", reason="proof", actor=ACTOR,
        run_id="run-proof-land", **_j(ctx))


async def confirm(ctx):
    await C.confirm_collection(ctx.queue, ctx.graph_id, _coll(DECLARED), ACTOR, **_j(ctx))


async def order(ctx):
    await C.set_collection_order(ctx.queue, ctx.graph_id, _coll(DECLARED), [SRC[1], SRC[0]], ACTOR, **_j(ctx))


async def file_targets(ctx):  # the hub's file gesture over two groups: ONE unit
    async with write_window(ctx.queue, ctx.graph_id, ctx.journal, unit=True):
        await C.refile_members(ctx.queue, ctx.graph_id, [SRC[0]], _coll(DECLARED), SERIES, ACTOR, **_j(ctx))
        await C.file_sources(ctx.queue, ctx.graph_id, SERIES, [SRC[1]], ACTOR, **_j(ctx))


async def rename(ctx):
    await C.rename_collection(ctx.queue, ctx.graph_id, _coll(SERIES), "Proof Series Renamed", ACTOR, **_j(ctx))


async def retire(ctx):
    await C.retire_collection(ctx.queue, ctx.graph_id, _coll(DECLARED), ACTOR, reason="proof", **_j(ctx))


async def urls(ctx):
    await C.bind_source_urls(ctx.queue, ctx.graph_id, [{"source_id": SRC[0], "url": "https://example.org/a"}],
                             ACTOR, **_j(ctx))


async def dates(ctx):
    await C.bind_source_dates(ctx.queue, ctx.graph_id, [{"source_id": SRC[0], "recorded_at": "2024-04-25"}],
                              ACTOR, **_j(ctx))


async def structure(ctx):
    await C.declare_structure(ctx.queue, ctx.graph_id,
                              [{"source_id": SRC[1], "structure": {"kind": "chapter", "chapter": 2},
                                "evidence": ["toc"]}], ACTOR, **_j(ctx))


async def reference(ctx):
    await C.add_reference(ctx.queue, ctx.graph_id, SRC[0], label="Proof ref", url="https://example.org/r",
                          actor=ACTOR, **_j(ctx))


async def reference_retract(ctx):
    ref = ctx.record("reference", "collection-curation")["wires"]["nodes"][0]["id"]
    await C.retract_reference(ctx.queue, ctx.graph_id, ref, actor=ACTOR, **_j(ctx))


CURATE = ("collection-curation",)
STEPS = [
    ProofStep("emit_a", emit_a, ("source-emission", "derivation"), retried=True),
    ProofStep("emit_a_again", emit_a, ()),  # a re-emission over what is there: no change
    ProofStep("emit_b", emit_b, ("source-emission", "derivation"), retried=True),
    ProofStep("declare", declare, ("collection-declaration",), retried=True),
    ProofStep("land", land, ("transcript-landing",), retried=True),
    ProofStep("confirm", confirm, CURATE),
    ProofStep("order", order, CURATE),
    ProofStep("file_targets", file_targets, ("collection-curation", "collection-declaration"), unit=True),
    ProofStep("rename", rename, CURATE),
    ProofStep("retire", retire, CURATE),
    ProofStep("urls", urls, CURATE),
    ProofStep("dates", dates, CURATE),
    ProofStep("structure", structure, CURATE),
    ProofStep("reference", reference, CURATE),
    ProofStep("reference_retract", reference_retract, CURATE),
]


@pytest.fixture(scope="module")
def uncrashed(tmp_path_factory):
    return run_proof(tmp_path_factory.mktemp("uncrashed"), MANIFESTS, STEPS)


def test_every_transcription_verb_is_proven_by_a_step(uncrashed):
    from cjm_transcription_core.emission import transcription_replay_handlers
    assert {v for s in STEPS for v in s.ops} == set(transcription_replay_handlers())
    assert all(c for s, c in zip(STEPS, uncrashed["counts"]) if s.ops)


# FINDING 249298fd: an emission's Derivation is planned from what THIS run added, so a crash between
# the emission's op and its Derivation loses the Derivation (the re-run adds nothing) -- strict, so
# the fix flips these to failures and their marks come off
LOST_DERIVATION = {("after_append", 1), ("after_apply", 1)}


@pytest.mark.parametrize("variant", [
    pytest.param(v, marks=pytest.mark.xfail(strict=True, reason="FINDING 249298fd")) if v in LOST_DERIVATION
    else v for v in crash_variants(STEPS)], ids=lambda v: f"{v[0]}@{v[1]}")
def test_a_crashed_verb_heals_to_the_uncrashed_run(tmp_path, uncrashed, variant):
    crashed = run_proof(tmp_path, MANIFESTS, STEPS, variant)
    assert crashed["counts"] == uncrashed["counts"]
    assert crashed["journal"] == uncrashed["journal"]
