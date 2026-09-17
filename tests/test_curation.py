"""Collection reads from the Source side (DEC 774dbe40: the assign-lane picker's
collection tier): `holding_collections` is one PART_OF-in query, `sibling_sources`
unions the live holding collections' members and drops the source itself and
every retired collection (ruling a7617bd4)."""
import asyncio
from types import SimpleNamespace

from cjm_substrate.core.queue import JobStatus
from cjm_transcription_core import curation


class FakeQueue:
    """Answers query_nodes from a canned table keyed by (label, far-end node id)."""

    def __init__(self, answers):
        self.answers = answers
        self.queries = []

    async def submit(self, graph_id, **kw):
        self.queries.append(kw)
        return str(len(self.queries))

    async def wait_for_job(self, jid):
        kw = self.queries[int(jid) - 1]
        q = kw["query"]
        rel = q.get("related") or {}
        rows = self.answers.get((q["label"], rel.get("direction"), rel.get("node_id")), [])
        return SimpleNamespace(status=JobStatus.completed,
                               result=SimpleNamespace(rows=rows), error=None)


def test_holding_collections_reads_part_of_in():
    q = FakeQueue({("Collection", "in", "src-1"): [
        {"id": "c-gpu", "title": "GPU MODE", "status": "confirmed"},
        {"id": "c-old", "title": "GPU MODE_OLD", "status": "retired"},
    ]})
    out = asyncio.run(curation.holding_collections(q, "g", "src-1"))
    assert [c["id"] for c in out] == ["c-gpu", "c-old"]
    assert out[1]["status"] == "retired"
    rel = q.queries[0]["query"]["related"]
    assert rel["relation_type"] == "PART_OF" and rel["direction"] == "in"
    assert rel["node_id"] == "src-1"


def test_sibling_sources_unions_live_collections_and_drops_self_and_retired():
    q = FakeQueue({
        ("Collection", "in", "src-1"): [
            {"id": "c-gpu", "title": "GPU MODE", "status": "confirmed"},
            {"id": "c-streams", "title": "GPU MODE Streams", "status": "confirmed"},
            {"id": "c-old", "title": "GPU MODE_OLD", "status": "retired"},
        ],
        ("Source", "out", "c-gpu"): [{"id": "src-1", "title": "Bonus"},
                                     {"id": "src-2", "title": "Lecture 1"}],
        ("Source", "out", "c-streams"): [{"id": "src-2", "title": "Lecture 1"},
                                         {"id": "src-3", "title": "Stream 4"}],
        ("Source", "out", "c-old"): [{"id": "src-9", "title": "old lecture"}],
    })
    out = asyncio.run(curation.sibling_sources(q, "g", "src-1"))
    assert [c["id"] for c in out["collections"]] == ["c-gpu", "c-streams"]
    assert out["siblings"] == {"src-2": "Lecture 1", "src-3": "Stream 4"}
    # the retired collection's members are never even read
    assert all(kw["query"].get("related", {}).get("node_id") != "c-old" for kw in q.queries)


def test_sibling_sources_unfiled_source_is_empty():
    q = FakeQueue({})
    out = asyncio.run(curation.sibling_sources(q, "g", "src-lonely"))
    assert out == {"collections": [], "siblings": {}}
