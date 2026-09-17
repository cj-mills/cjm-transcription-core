"""The escalation prompt's LIVE slot-text provider (finding c63cd2e3, user ruling
2026-09-17): a caller holding the corrected spine fills BEFORE / AFTER / DRAFT from
it; the manifest answers only where the provider returns nothing; the template hash
never moves."""
from cjm_transcription_core.chunk import DEFAULT_ESCALATION_PROMPT, prompt_hash_of, render_escalation_prompt


def _manifest():
    def seg(i, start, end, text):
        return {"index": i, "start": start, "end": end,
                "transcripts": {"whisper": {"text": text}}}
    return {"config": {"transcriber_capabilities": ["whisper"]},
            "collections": [{"title": "GPU MODE"}],
            "sources": [{"source_path": "x/Lecture.mp4",
                         "segments": [seg(0, 0, 100, "raw zero"), seg(1, 100, 200, "raw one"),
                                      seg(2, 200, 300, "raw two")]}]}


def test_provider_fills_all_three_slots_and_names_their_sources():
    calls = []

    def live(start, end):
        calls.append((start, end))
        return {0.0: "CORRECTED zero", 100.0: "CORRECTED one", 200.0: "CORRECTED two"}[start]

    out = render_escalation_prompt(_manifest(), 0, 1, slot_text=live)
    assert calls == [(0.0, 100.0), (200.0, 300.0), (100.0, 200.0)]   # prev, next, self — by chunk span
    assert out["slots"]["prev_text"] == "CORRECTED zero"
    assert out["slots"]["next_text"] == "CORRECTED two"
    assert out["slots"]["draft_text"] == "CORRECTED one"          # the draft reads the spine too
    assert out["slot_sources"] == {"prev_text": "spine", "next_text": "spine", "draft_text": "spine"}
    assert out["prompt_hash"] == prompt_hash_of(DEFAULT_ESCALATION_PROMPT)   # slot text never rides the identity


def test_provider_gaps_fall_back_to_the_manifest_per_slot():
    out = render_escalation_prompt(_manifest(), 0, 1,
                                   slot_text=lambda s, e: "   " if s == 0.0 else (None if s == 200.0 else "LIVE one"))
    assert out["slots"]["prev_text"] == "raw zero" and out["slot_sources"]["prev_text"] == "manifest"
    assert out["slots"]["next_text"] == "raw two" and out["slot_sources"]["next_text"] == "manifest"
    assert out["slots"]["draft_text"] == "LIVE one" and out["slot_sources"]["draft_text"] == "spine"


def test_no_provider_is_the_manifest_path_and_edges_read_none():
    out = render_escalation_prompt(_manifest(), 0, 0)
    assert out["slots"]["prev_text"] == "(none)" and out["slot_sources"]["prev_text"] == "none"
    assert out["slot_sources"]["draft_text"] == "manifest"
    # a first chunk with a provider: no corrected neighbour before it, its own corrected draft
    first = render_escalation_prompt(_manifest(), 0, 0, slot_text=lambda s, e: f"LIVE {s:.0f}")
    assert first["slots"]["prev_text"] == "(none)" and first["slots"]["draft_text"] == "LIVE 0"
    # the provider's text still honours the char budgets
    long = render_escalation_prompt(_manifest(), 0, 1, slot_text=lambda s, e: "x" * 5000, draft_chars=10,
                                    neighbour_chars=4)
    assert len(long["slots"]["draft_text"]) == 10 and len(long["slots"]["prev_text"]) == 4
