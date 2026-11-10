from app.agenteval import (
    count_flags,
    is_daily_limit,
    is_refusal,
    lie_detected,
    lie_removed,
    parse_wait,
    render_markdown,
    summarize,
)


def test_parse_wait_reads_seconds_minutes_and_milliseconds():
    assert parse_wait("Rate limit. Please try again in 7.5s.") == 7.5
    assert parse_wait("Please try again in 1m12.5s") == 72.5
    assert parse_wait("try again in 250ms") == 0.25
    assert parse_wait("no hint here") is None


def test_daily_limit_detection():
    assert is_daily_limit("Rate limit reached ... on tokens per day (TPD): Limit 100000")
    assert is_daily_limit("Generate content requests per day limit exceeded on Free Tier")
    assert not is_daily_limit("Rate limit reached on tokens per minute (TPM). try again in 5s")


def test_refusal_detection_ignores_case_and_spacing():
    assert is_refusal("I could not find this in the document.")
    assert is_refusal("  i could   not find this in the document ")
    assert not is_refusal("Revenue was 22,465 million.")


def test_count_flags():
    v = {"sentences": [{"verdict": "supported"}, {"verdict": "partial"}, {"verdict": "unsupported"}]}
    assert count_flags(v) == (3, 2)
    assert count_flags({"sentences": []}) == (0, 0)


def test_lie_detection_and_removal():
    flagged = {"sentences": [
        {"text": "Revenue was 22,465 million.", "verdict": "supported"},
        {"text": "The company was founded on Mars in 1850.", "verdict": "unsupported"},
    ]}
    missed = {"sentences": [{"text": "The company was founded on Mars in 1850.", "verdict": "supported"}]}
    assert lie_detected(flagged)
    assert not lie_detected(missed)
    assert lie_removed("Revenue was 22,465 million.")
    assert not lie_removed("Revenue rose. The company was founded on Mars.")


def _rec(**kw):
    base = {
        "kind": "normal", "answerable": True, "first_sentences": 4, "first_flagged": 0,
        "final_sentences": 4, "final_flagged": 0, "first_trust": 1.0, "final_trust": 1.0,
        "rewrites": 0, "refused": False, "latency_s": 10.0,
    }
    base.update(kw)
    return base


def test_summarize_computes_rates():
    records = [
        _rec(correct="correct", first_correct="correct", retrieval_rank=1,
             first_flagged=1, first_trust=0.75, rewrites=1),
        _rec(correct="incorrect", first_correct="incorrect", retrieval_rank=None, refused=True,
             first_sentences=0, final_sentences=0),
        _rec(answerable=False, refused=True, first_sentences=0, final_sentences=0),
        _rec(answerable=False, refused=False, first_sentences=2, final_sentences=2),
        _rec(kind="plant", lie_detected=True, lie_removed=True, latency_s=20.0),
        _rec(kind="plant", lie_detected=False, lie_removed=False, latency_s=30.0),
    ]
    s = summarize(records)
    assert (s["answerable_n"], s["unanswerable_n"], s["plant_n"]) == (2, 2, 2)
    assert s["correct_rate"] == 0.5 and s["incorrect_rate"] == 0.5
    assert s["false_refusal_rate"] == 0.5
    assert s["refusal_accuracy"] == 0.5
    assert s["retrieval_hit_rate"] == 0.5
    assert s["first_flagged_rate"] == round(1 / 6, 4)  # 1 flagged out of 4 + 0 + 0 + 2 sentences
    assert s["final_flagged_rate"] == 0.0
    assert s["rewrite_rate"] == 0.25
    assert s["lie_detected_rate"] == 0.5 and s["lie_removed_rate"] == 0.5
    assert s["p95_latency_s"] == 30.0


def test_summarize_handles_empty_groups():
    s = summarize([_rec(kind="plant", lie_detected=True, lie_removed=True)])
    assert s["correct_rate"] is None
    assert s["refusal_accuracy"] is None


def test_markdown_contains_key_numbers():
    s = summarize([_rec(correct="correct", first_correct="correct", retrieval_rank=1)])
    meta = {"run_name": "t", "mode": "hybrid_rerank", "k": 5, "llm": "x", "date": "2026-11-10"}
    text = render_markdown(s, meta)
    assert "| Metric | Result |" in text
    assert "100% / 0% / 0%" in text