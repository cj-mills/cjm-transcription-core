"""Verbatim-prompt bench — voxtral instruct-mode variant vs the transcription-template baseline (dc9ecf0a).

Scores the prompt_mode="instruct" variant against the TRANSCRIPTION-template
baseline over the corrected head of "How I use LLMs" (the walk's 551-unit
frontier, 9e9ea090): the user's hand-restored hesitation markers ARE the
ground truth, so the bench is free. Primary metric = marker retention
(recall of GT hesitation-marker tokens under token alignment) + marker
precision (over-generation guard). Regression surfaces ride along: token
error rate, immediate-word-repeat retention, 'you know' bigram retention,
lowercase-I orthographic drift (dc31c33c item 4), unaligned candidate tail
(hallucination class 21b778e3), and substitution samples for eyeballing the
homophone/orthographic classes (02e67e65).

Three phases, each owning-env bound (env truth: no shared env exists):
  extract     correction-core env — offline effective-spine projection
              (direct read-only sqlite + the core's PURE projection fns;
              validated: reproduces the live 4517-unit spine exactly),
              GT + baseline text per covered aseg -> extract.json
  transcribe  test-voxtral-hf WORKER env — instruct-mode voxtral over the
              covered asegs' model-input wavs -> variant.json
  score       any env (stdlib only) — the report -> report.json + stdout
  all         subprocess the three phases through their envs

Baseline aseg texts are NOT re-transcribed: the canonical db's voxtral
Transcript nodes are the deterministic greedy-decode baseline already.
The canonical graph db is opened read-only; outputs land under
runs/verbatim_bench/.

Run:  python tests_manual/bench_voxtral_verbatim_prompt.py all
"""

import argparse
import difflib
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

BASE = Path("/mnt/SN850X_8TB_EXT4/Projects/GitHub/cj-mills")
TRANSCRIPTION = BASE / "cjm-transcription-core"
DB = TRANSCRIPTION / ".cjm/data/cjm-capability-graph-sqlite/context_graph.db"
CORRECTION_PY = Path.home() / "miniforge3/envs/cjm-transcript-correction-core/bin/python"
WORKER_PY = TRANSCRIPTION / "runtime/envs/test-voxtral-hf/bin/python"  # capability worker env (env-truth 4th sighting)
HF_HOME = "/mnt/SN850X_8TB_EXT4/Model_Cache/huggingface"
OUT_DIR = TRANSCRIPTION / "runs/verbatim_bench"

DEFAULT_SOURCE = "11a9a3da"  # How I use LLMs (id prefix; --source overrides — any corrected source benches)
DEFAULT_HEAD = 551           # its corrected walk frontier in effective units (9e9ea090); other sources default to the full spine
VOXTRAL = "cjm-capability-voxtral-hf"

# Hesitation-marker lexicon (normalized-token space). Deliberately UNIGRAM +
# hyphen-joined forms; 'you know' is scored separately as a bigram because its
# tokens are also ordinary speech.
MARKERS = {"um", "uh", "hmm", "mhm", "mm", "mm-hmm", "uh-huh", "er", "erm", "ah", "huh"}


def tokenize(text: str):  # -> (raw_tokens, norm_tokens)
    """Whitespace-split, then per-token strip of surrounding punctuation.

    norm = lowercased, outer punctuation stripped (internal hyphens/apostrophes
    kept so 'mm-hmm' and contractions survive as single tokens); raw keeps the
    original casing for orthographic checks (the lowercase-I drift count)."""
    raw = text.split()
    norm = []
    for t in raw:
        n = re.sub(r"^[^\w]+|[^\w]+$", "", t.lower())
        norm.append(n)
    pairs = [(r, n) for r, n in zip(raw, norm) if n]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def score_candidate(gt_text: str, cand_text: str):  # -> metric dict
    """Token-alignment scoring of one candidate transcript against GT.

    difflib.SequenceMatcher over normalized token lists; a GT token is
    RETAINED when it sits inside an equal block. Marker recall is the
    experiment's primary metric; TER (diff-based token error rate, an
    LCS approximation, comparable across candidates) + repeat/'you know'
    retention + orthographic/hallucination surfaces guard the regressions."""
    gt_raw, gt = tokenize(gt_text)
    c_raw, c = tokenize(cand_text)
    sm = difflib.SequenceMatcher(None, gt, c, autojunk=False)
    ops = sm.get_opcodes()
    gt_eq, c_eq = set(), set()
    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal":
            gt_eq.update(range(i1, i2))
            c_eq.update(range(j1, j2))

    gt_marks = [i for i, t in enumerate(gt) if t in MARKERS]
    c_marks = [j for j, t in enumerate(c) if t in MARKERS]
    kept = [i for i in gt_marks if i in gt_eq]
    lost = Counter(gt[i] for i in gt_marks if i not in gt_eq)

    yk = [i for i in range(len(gt) - 1) if gt[i] == "you" and gt[i + 1] == "know"]
    yk_kept = [i for i in yk if i in gt_eq and i + 1 in gt_eq]
    reps = [i for i in range(len(gt) - 1) if gt[i] == gt[i + 1]]
    reps_kept = [i for i in reps if i in gt_eq and i + 1 in gt_eq]

    errs = sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in ops if tag != "equal")
    last_c_eq = max(c_eq) if c_eq else -1
    subs = []
    for tag, i1, i2, j1, j2 in ops:
        if tag == "replace" and len(subs) < 15:
            subs.append({"gt": " ".join(gt_raw[i1:i2])[:80], "cand": " ".join(c_raw[j1:j2])[:80]})

    return {
        "gt_tokens": len(gt), "cand_tokens": len(c),
        "marker_gt": len(gt_marks), "marker_kept": len(kept),
        "marker_recall": round(len(kept) / len(gt_marks), 4) if gt_marks else None,
        "marker_cand": len(c_marks),
        "marker_precision": round(len([j for j in c_marks if j in c_eq]) / len(c_marks), 4) if c_marks else None,
        "marker_lost_by_token": dict(lost),
        "you_know_gt": len(yk), "you_know_kept": len(yk_kept),
        "repeats_gt": len(reps), "repeats_kept": len(reps_kept),
        "ter": round(errs / len(gt), 4) if gt else None,
        "lower_i": sum(1 for t in c_raw if t == "i"),
        "lower_i_gt": sum(1 for t in gt_raw if t == "i"),
        "cand_tail_unaligned": len(c) - 1 - last_c_eq,
        "substitution_samples": subs,
    }


def cmd_extract(args):
    """GT + baseline per covered aseg (RUN IN the correction-core env).

    Offline effective-spine projection: direct READ-ONLY sqlite loads feed the
    core's pure project_effective_spine (queue-free — the loading is the only
    queue-backed part of the live path). Validated against the live spine:
    4517 effective units on source 11a9a3da, matching the walk's N/4517
    frontier display. GT per aseg = index-ordered effective-unit texts assigned
    by time midpoint (wordless inserts contribute empty text and vanish in the
    join; within-aseg boundary shifts cannot move text across aseg
    concatenation boundaries). Baseline = the canonical db's deterministic
    voxtral Transcript per aseg. --head N bounds GT to the walk frontier
    (0 = the whole spine is corrected)."""
    from cjm_transcript_correction_core.graph import project_effective_spine
    from cjm_transcript_correction_core.models import SpineSegment

    src_id, title = resolve_source(args.source)
    skeleton = resolve_skeleton(src_id, args.skeleton)
    print(f"source {src_id[:8]} ({title}) spine {skeleton[:16]}…")

    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    segs = []
    rend_ids = set()  # the renditions THIS skeleton's spine hangs under
    for r in con.execute(
            "SELECT id, properties FROM nodes WHERE label='Segment' "
            "AND json_extract(properties,'$.source_id')=? "
            "AND json_extract(properties,'$.skeleton_hash')=?", (src_id, skeleton)):
        p = json.loads(r["properties"])
        rend_ids.add(p.get("rendition_id"))
        segs.append(SpineSegment(
            id=r["id"], index=int(p["index"]), text=p.get("text") or "",
            start_time=p.get("start_time"), end_time=p.get("end_time"),
            source_locator=None, content_hash=None,
            text_from=p.get("text_from"), text_slices=[]))
    segs.sort(key=lambda s: s.index)

    corrections = []
    for r in con.execute(
            "SELECT id, properties FROM nodes WHERE label='Correction' "
            "AND json_extract(properties,'$.payload.source_id')=?", (src_id,)):
        p = json.loads(r["properties"])
        p["id"] = r["id"]
        corrections.append(p)
    ids = {c["id"] for c in corrections}
    superseded = set()
    for r in con.execute("SELECT source_id, target_id FROM edges WHERE relation_type='SUPERSEDES'"):
        if r["source_id"] in ids or r["target_id"] in ids:
            superseded.add(r["target_id"])
    active = [c for c in corrections
              if c["id"] not in superseded and c.get("status") != "proposed"]

    eff = project_effective_spine(segs, active)
    head = args.head or len(eff)
    print(f"layer-0 {len(segs)} segs + {len(active)} active corrections "
          f"-> {len(eff)} effective units (GT head {head})")
    cutoff = eff[head - 1].end_time

    asegs = []
    for r in con.execute(
            "SELECT n.id, n.properties FROM nodes n JOIN edges e ON e.source_id=n.id "
            "WHERE n.label='AudioSegment' AND e.relation_type='PART_OF' AND e.target_id=?", (src_id,)):
        p = json.loads(r["properties"])
        asegs.append({"id": r["id"], "index": p["index"], "start": p["start"], "end": p["end"]})
    asegs.sort(key=lambda a: a["index"])
    covered = [a for a in asegs if a["end"] <= cutoff + 0.05]
    print(f"cutoff t={cutoff:.1f}s -> {len(covered)} complete asegs of {len(asegs)}")

    out_asegs = []
    for a in covered:
        rend = None
        for r in con.execute(
                "SELECT id, properties FROM nodes WHERE label='AudioRendition' "
                "AND json_extract(properties,'$.audio_segment_id')=?", (a["id"],)):
            if r["id"] in rend_ids:
                rend = {"id": r["id"], **json.loads(r["properties"])}
        assert rend, f"no spine rendition for aseg {a['index']}"
        base = con.execute(
            "SELECT n.properties FROM nodes n JOIN edges e ON e.source_id=n.id "
            "WHERE n.label='Transcript' AND e.relation_type='DERIVED_FROM' AND e.target_id=? "
            "AND json_extract(n.properties,'$.transcriber')=?", (rend["id"], VOXTRAL)).fetchone()
        assert base, f"no baseline voxtral Transcript for aseg {a['index']}"
        units = [u for u in eff
                 if u.start_time is not None and u.end_time is not None
                 and a["start"] <= (u.start_time + u.end_time) / 2 < a["end"]]
        gt = " ".join(u.text.strip() for u in units if u.text and u.text.strip())
        out_asegs.append({
            "index": a["index"], "start": a["start"], "end": a["end"],
            "model_input_path": rend["model_input_path"],
            "gt_text": gt,
            "baseline_text": json.loads(base["properties"])["text"],
        })
        print(f"  aseg {a['index']}: {len(units)} units, GT {len(gt)} chars")

    d = bench_dir(src_id)
    (d / "extract.json").write_text(json.dumps({
        "source_id": src_id, "title": title, "skeleton": skeleton, "head": head,
        "cutoff": cutoff, "effective_units": len(eff), "asegs": out_asegs}, indent=1))
    print(f"wrote {d / 'extract.json'}")


def cmd_transcribe(args):
    """Instruct-mode voxtral over the covered asegs (RUN IN the test-voxtral-hf worker env).

    One model load, greedy decode (config defaults), prompt_mode='instruct'.
    --prompt-file overrides the config's default instruct_prompt (prompt
    A/B lane); --tag names the run's output files so variants coexist. The
    recorded config + per-result metadata land in the variant json so every
    run is attributable to its exact prompt."""
    from cjm_capability_voxtral_hf.capability import VoxtralHFCapability
    d = bench_dir(resolve_source(args.source)[0])
    data = json.loads((d / "extract.json").read_text())
    cfg = {"prompt_mode": "instruct"}
    if args.prompt_file:
        cfg["instruct_prompt"] = Path(args.prompt_file).read_text().strip()
    cap = VoxtralHFCapability()
    cap.initialize(cfg)
    rows = []
    for a in data["asegs"]:
        t0 = time.monotonic()
        res = cap.transcribe(a["model_input_path"])
        wall = round(time.monotonic() - t0, 1)
        rows.append({"index": a["index"], "text": res.text,
                     "wall_s": wall, "metadata": res.metadata})
        print(f"aseg {a['index']}: {len(res.text)} chars in {wall}s")
    name = f"variant_{args.tag}.json" if args.tag else "variant.json"
    (d / name).write_text(json.dumps(
        {"config": cap.get_current_config(), "asegs": rows}, indent=1))
    print(f"wrote {d / name}")


def cmd_score(args):
    """Score baseline + every present variant against GT, per aseg and pooled (stdlib only).

    Candidates = baseline + variant.json (default prompt) + any variant_<tag>.json
    present in the source's bench dir — one table, all prompts side by side."""
    d = bench_dir(resolve_source(args.source)[0])
    ex = json.loads((d / "extract.json").read_text())

    cands = {}
    for f in sorted(d.glob("variant*.json")):
        name = f.stem.replace("variant_", "") if f.stem != "variant" else "variant"
        v = json.loads(f.read_text())
        cands[name] = {a["index"]: a["text"] for a in v["asegs"]}

    report = {"source_id": ex["source_id"], "title": ex.get("title"),
              "per_aseg": [], "pooled": {}}
    for a in ex["asegs"]:
        row = {"index": a["index"],
               "baseline": score_candidate(a["gt_text"], a["baseline_text"])}
        for name, texts in cands.items():
            row[name] = score_candidate(a["gt_text"], texts[a["index"]])
        report["per_aseg"].append(row)
    gt_all = " ".join(a["gt_text"] for a in ex["asegs"])
    pooled = {"baseline": score_candidate(gt_all, " ".join(a["baseline_text"] for a in ex["asegs"]))}
    for name, texts in cands.items():
        pooled[name] = score_candidate(gt_all, " ".join(texts[a["index"]] for a in ex["asegs"]))
    report["pooled"] = pooled

    names = ["baseline"] + list(cands)
    width = max(len(n) for n in names) + 1
    print(f"{'aseg':>6} {'cand':<{width}} {'mk_gt':>5} {'mk_kept':>7} {'recall':>7} "
          f"{'prec':>6} {'yk':>7} {'reps':>7} {'ter':>6} {'low_i':>5} {'tail':>5}")
    rows = [(str(r["index"]), r) for r in report["per_aseg"]] + [("POOLED", report["pooled"])]
    for label, r in rows:
        for cand in names:
            m = r[cand]
            print(f"{label:>6} {cand:<{width}} {m['marker_gt']:>5} {m['marker_kept']:>7} "
                  f"{str(m['marker_recall']):>7} {str(m['marker_precision']):>6} "
                  f"{m['you_know_kept']:>3}/{m['you_know_gt']:<3} "
                  f"{m['repeats_kept']:>3}/{m['repeats_gt']:<3} "
                  f"{str(m['ter']):>6} {m['lower_i']:>5} {m['cand_tail_unaligned']:>5}")
    for cand in names:
        m = report["pooled"][cand]
        print(f"\n{cand} pooled: lost markers {m['marker_lost_by_token']}; sub samples:")
        for s in m["substitution_samples"][:6]:
            print(f"  GT: {s['gt']!r}  ->  {s['cand']!r}")

    (d / "report.json").write_text(json.dumps(report, indent=1))
    print(f"\nwrote {d / 'report.json'}")


def cmd_all(args):
    """Chain extract -> transcribe -> score, each through its owning env."""
    me = Path(__file__).resolve()
    env = dict(os.environ, HF_HOME=HF_HOME, CUDA_VISIBLE_DEVICES="0")
    passthru = ["--source", args.source, "--head", str(args.head or 0)]
    if args.skeleton:
        passthru += ["--skeleton", args.skeleton]
    if args.prompt_file:
        passthru += ["--prompt-file", args.prompt_file]
    if args.tag:
        passthru += ["--tag", args.tag]
    for py, phase in ((CORRECTION_PY, "extract"), (WORKER_PY, "transcribe"), (sys.executable, "score")):
        print(f"== {phase} ({py}) ==")
        proc = subprocess.run([str(py), str(me), phase] + passthru, env=env)
        if proc.returncode != 0:
            raise SystemExit(f"{phase} failed rc={proc.returncode}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("phase", choices=["extract", "transcribe", "score", "all"])
    ap.add_argument("--source", default=DEFAULT_SOURCE,
                    help="Source id prefix or title substring (default: How I use LLMs)")
    ap.add_argument("--skeleton", default="",
                    help="skeleton hash prefix (default: the TUI sidecar's chosen spine)")
    ap.add_argument("--head", type=int, default=None,
                    help="GT frontier in effective units; 0 = whole spine corrected "
                         "(default: the walk frontier for the default source, else 0)")
    ap.add_argument("--prompt-file", default="",
                    help="file whose text overrides the default instruct_prompt (prompt A/B lane)")
    ap.add_argument("--tag", default="",
                    help="names this run's variant_<tag>.json so prompt variants coexist")
    args = ap.parse_args()
    if args.head is None:
        args.head = DEFAULT_HEAD if args.source == DEFAULT_SOURCE else 0
    {"extract": cmd_extract, "transcribe": cmd_transcribe,
     "score": cmd_score, "all": cmd_all}[args.phase](args)


# NOTE: the __main__ entry must stay the LAST region — add-symbol appends at
# module tail, and canonical emit follows SLOT order (a file-level reorder is
# reverted by the next authored write), so the guard lives in the final region
# below; new symbols must be added BEFORE it or re-relocated (ba810a2a).


def resolve_source(selector: str):  # -> (source_id, title)
    """Resolve a Source by id prefix, else case-insensitive title substring (loud on ambiguity)."""
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    rows = [(r[0], json.loads(r[1]).get("title") or "") for r in con.execute(
        "SELECT id, properties FROM nodes WHERE label='Source'")]
    con.close()
    hits = ([r for r in rows if r[0].startswith(selector)]
            or [r for r in rows if selector.lower() in r[1].lower()])
    if len(hits) != 1:
        raise SystemExit(f"source selector {selector!r} matched {len(hits)}: {[t for _, t in hits]}")
    return hits[0]


def resolve_skeleton(source_id: str, arg: str):  # -> skeleton hash
    """The bench spine: --skeleton hash prefix, else the TUI sidecar's CHOSEN
    skeleton for the source (the spine the user actually walked — view state,
    read from the db-adjacent .tui-state.json sidecar)."""
    if arg:
        con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
        hashes = {json.loads(r[0]).get("skeleton_hash") for r in con.execute(
            "SELECT properties FROM nodes WHERE label='Segment' "
            "AND json_extract(properties,'$.source_id')=?", (source_id,))}
        con.close()
        hits = [h for h in hashes
                if h and (h.startswith(arg) or h.split(":", 1)[-1].startswith(arg))]
        if len(hits) != 1:
            raise SystemExit(f"skeleton prefix {arg!r} matched {len(hits)}: {hits}")
        return hits[0]
    state = json.loads(Path(f"{DB}.tui-state.json").read_text())
    sk = (state.get(source_id) or {}).get("skeleton")
    if not sk:
        raise SystemExit(f"no --skeleton given and no sidecar spine choice for source {source_id[:8]}")
    return sk


def bench_dir(source_id: str) -> Path:  # per-source output dir under runs/verbatim_bench/
    d = OUT_DIR / source_id[:8]
    d.mkdir(parents=True, exist_ok=True)
    return d


if __name__ == "__main__":  # CLI entry — must stay the LAST region (see note above)
    main()
