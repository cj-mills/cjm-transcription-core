"""Past-run results for the setup TUI: the core's own runs/*.json manifests read
into browsable state, plus the source-browser affordances they enable (work
item 6768bafb — results view, prior-run chips, content hash-check).

Pure logic, Textual-free (sources.py precedent: everything below the paint path
tests directly; the app only paints it). The manifests are the headless core's
OWN RunManifest jsons — the TUI reads what the run wrote, no parallel record —
and the default "runs" dir is the core CLI's own cwd-relative default, so the
TUI sees exactly the runs its hand-offs produced. Content identity reuses the
substrate's hash_file (the Source-node identity input), so a hash-check match
here means what emission meant by it."""

import json
from pathlib import Path
from typing import Any, Dict, List

from cjm_substrate.core.workspace import resolve_recorded_tree
from cjm_substrate.utils.hashing import hash_file


class RunIndex:
    """runs/*.json manifests loaded newest-first + the lookups the TUI paints from.

    load() tolerates unreadable/foreign jsons (skipped, never raises) — the runs
    dir is shared ground and one corrupt file must not hide the rest.
    counts_by_path() powers the browser's prior-run chips WITHOUT hashing
    anything (resolved-path keyed, cheap enough to rebuild per stage entry);
    hash_check() is the on-demand content-identity match that catches the
    renamed/moved/duplicated sources path matching misses (drive-1 follow-up:
    content hash IS the Source identity, so a match here is a true prior run
    of this AUDIO whatever the file is called now)."""

    def __init__(self, runs_dir: str = "runs"):  # The core CLI's cwd-relative default
        self.runs_dir = Path(runs_dir)
        self.runs: List[Dict[str, Any]] = []  # Manifest dicts, newest first (+ "_path")

    def load(self) -> int:  # Number of manifests loaded
        """(Re)read every readable run manifest in runs_dir, newest first."""
        rows: List[Dict[str, Any]] = []
        try:
            files = sorted(self.runs_dir.glob("*.json"))
        except OSError:
            files = []
        for f in files:
            try:
                # ${WS}/ recorded paths (5daadfc4 rung f) resolve at load,
                # anchored at the manifest's own location.
                m = resolve_recorded_tree(json.loads(f.read_text()), f)
            except (OSError, ValueError):
                continue
            if not (isinstance(m, dict) and m.get("run_id")
                    and isinstance(m.get("sources"), list)):
                continue  # foreign json in the runs dir — not ours
            if "transcription-core" not in str(m.get("format", "")):
                # BOTH cores write run_id+sources manifests into the same
                # runs/ default — the format tag is what separates a
                # transcription run from a decomp run (finding 5329fbd8).
                continue
            m["_path"] = str(f)
            rows.append(m)
        rows.sort(key=lambda m: float(m.get("created_at") or 0.0), reverse=True)
        self.runs = rows
        return len(rows)

    def transcribers(self, manifest: Dict[str, Any]) -> List[str]:
        """The run's transcriber instance ids (config snapshot; [] when absent)."""
        cfg = manifest.get("config") or {}
        ids = cfg.get("transcriber_capabilities") or []
        return [str(i) for i in ids] if isinstance(ids, list) else []

    def collection_titles(self) -> List[str]:
        """Distinct collection titles from past manifests, newest-first (0.4.0
        `collections`) — the sources stage's cheap pick-existing surface:
        title identity means retyping one of these IS selecting the existing
        collection node (ae3464fc), no graph stack needed at setup time."""
        seen: List[str] = []
        for m in self.runs:
            for c in (m.get("collections") or []):
                t = str(c.get("title") or "").strip()
                if t and t not in seen:
                    seen.append(t)
        return seen

    def counts_by_path(self) -> Dict[str, int]:
        """resolved source_path -> number of runs that included it (path-keyed,
        NO hashing — the browse-time chip must stay free; hash truth is h)."""
        counts: Dict[str, int] = {}
        for m in self.runs:
            for s in m["sources"]:
                p = s.get("source_path")
                if p:
                    key = str(Path(p).resolve())
                    counts[key] = counts.get(key, 0) + 1
        return counts

    @staticmethod
    def dir_count(counts: Dict[str, int], dir_key: str) -> int:
        """Sources-with-runs under a directory (prefix sum over the counts map)."""
        prefix = dir_key.rstrip("/") + "/"
        return sum(n for p, n in counts.items() if p.startswith(prefix))

    def hash_check(self, path: str) -> Dict[str, Any]:
        """Hash the file with the substrate's hash_file and match manifests.

        Returns {"content_hash", "matches": [{"run_id", "source_path"}, ...]} —
        matches are CONTENT-identity hits, so a renamed/moved/duplicate source
        still finds its prior runs. May raise OSError (caller paints it)."""
        digest = hash_file(path)
        matches: List[Dict[str, Any]] = []
        for m in self.runs:
            for s in m["sources"]:
                if s.get("content_hash") and s["content_hash"] == digest:
                    matches.append({"run_id": m.get("run_id"),
                                    "source_path": s.get("source_path")})
        return {"content_hash": digest, "matches": matches}
