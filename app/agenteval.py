"""Helpers for the end-to-end agent evaluation.

Everything except judge_correctness() is plain Python, so it is unit-tested without any LLM.
"""
import math
import re
import statistics
from typing import Literal

from pydantic import BaseModel, Field

from app import config
from app.chat import get_chat_model

REFUSAL_PREFIX = "i could not find this in the document"


class CorrectnessResult(BaseModel):
    verdict: Literal["correct", "partial", "incorrect"]
    reason: str = Field(default="")


CORRECTNESS_RULES = """You grade an ANSWER against a trusted REFERENCE ANSWER for a question.
- correct: the answer states the key facts of the reference (numbers, names, dates and lists must match). Extra true detail is fine.
- partial: the answer has some of the key facts, or one detail differs, or a list is incomplete.
- incorrect: the answer misses the key facts, contradicts the reference, or says the information was not found.
Judge only against the reference. Ignore citation markers like [1].
Return ONLY a JSON object, for example: {"verdict": "correct", "reason": "short reason"}"""


# ---------- small pure helpers ----------

def is_refusal(text: str) -> bool:
    t = re.sub(r"\s+", " ", (text or "").strip().lower())
    return t.startswith(REFUSAL_PREFIX)


def count_flags(verification: dict) -> tuple[int, int]:
    """Return (number of sentences, number not fully supported)."""
    sentences = verification.get("sentences", [])
    flagged = sum(1 for s in sentences if s["verdict"] != "supported")
    return len(sentences), flagged


def lie_detected(first_verification: dict) -> bool:
    """True if the planted-lie sentence was NOT marked 'supported' by the first fact-check."""
    for s in first_verification.get("sentences", []):
        if re.search(r"\bmars\b", s["text"], re.I) and s["verdict"] != "supported":
            return True
    return False


def lie_removed(final_answer: str) -> bool:
    """True if the planted lie is gone from the final answer."""
    return not re.search(r"\bmars\b", final_answer or "", re.I)


def is_daily_limit(message: str) -> bool:
    """True for 'per day' quota errors, where waiting a few seconds will not help."""
    t = (message or "").lower()
    return "per day" in t or "(tpd)" in t or "(rpd)" in t


def parse_wait(message: str):
    """Read 'try again in 7.2s' / '1m12.5s' / '250ms' from a rate-limit message. Returns seconds or None."""
    m = re.search(r"try again in ([0-9hms.]+)", message or "")
    if not m:
        return None
    total = 0.0
    for value, unit in re.findall(r"(\d+(?:\.\d+)?)(ms|h|m|s)", m.group(1)):
        v = float(value)
        total += {"ms": v / 1000, "s": v, "m": v * 60, "h": v * 3600}[unit]
    return total or None


# ---------- LLM correctness judge ----------

def judge_correctness(question: str, reference: str, answer: str) -> CorrectnessResult:
    prompt = (
        f"{CORRECTNESS_RULES}\n\nQUESTION: {question}\n\n"
        f"REFERENCE ANSWER: {reference}\n\nANSWER TO GRADE: {answer}"
    )
    kwargs = {"method": "json_mode"} if config.LLM_PROVIDER == "groq" else {}
    judge = get_chat_model(0.0).with_structured_output(CorrectnessResult, **kwargs)
    last_error = None
    for _ in range(3):
        try:
            return judge.invoke(prompt)
        except Exception as e:
            if is_daily_limit(str(e)):
                raise
            last_error = e
    raise last_error


def grade(item: dict, answer: str) -> tuple[str, str]:
    """Return (verdict, reason). A refusal on an answerable question is simply 'incorrect'."""
    if is_refusal(answer):
        return "incorrect", "the system refused to answer"
    result = judge_correctness(item["question"], item["answer"], answer)
    return result.verdict, result.reason


# ---------- summary ----------

def _rate(count: int, total: int):
    return round(count / total, 4) if total else None


def _mean(values):
    return round(statistics.mean(values), 4) if values else None


def summarize(records: list[dict]) -> dict:
    normal = [r for r in records if r["kind"] == "normal"]
    answerable = [r for r in normal if r["answerable"]]
    unanswerable = [r for r in normal if not r["answerable"]]
    plant = [r for r in records if r["kind"] == "plant"]
    with_first = [r for r in normal if r["first_sentences"] > 0]
    with_final = [r for r in normal if r["final_sentences"] > 0]

    def share(items, predicate):
        return _rate(sum(1 for r in items if predicate(r)), len(items))

    latencies = sorted(r["latency_s"] for r in records)
    p95 = latencies[max(0, math.ceil(0.95 * len(latencies)) - 1)] if latencies else None

    return {
        "answerable_n": len(answerable),
        "unanswerable_n": len(unanswerable),
        "plant_n": len(plant),
        "correct_rate": share(answerable, lambda r: r.get("correct") == "correct"),
        "partial_rate": share(answerable, lambda r: r.get("correct") == "partial"),
        "incorrect_rate": share(answerable, lambda r: r.get("correct") == "incorrect"),
        "correct_rate_first_draft": share(answerable, lambda r: r.get("first_correct") == "correct"),
        "false_refusal_rate": share(answerable, lambda r: r["refused"]),
        "refusal_accuracy": share(unanswerable, lambda r: r["refused"]),
        "retrieval_hit_rate": share(answerable, lambda r: r.get("retrieval_rank") is not None),
        "first_flagged_rate": _rate(
            sum(r["first_flagged"] for r in normal), sum(r["first_sentences"] for r in normal)
        ),
        "final_flagged_rate": _rate(
            sum(r["final_flagged"] for r in normal), sum(r["final_sentences"] for r in normal)
        ),
        "first_answers_with_flags": share(with_first, lambda r: r["first_flagged"] > 0),
        "final_answers_with_flags": share(with_final, lambda r: r["final_flagged"] > 0),
        "mean_trust_first": _mean([r["first_trust"] for r in with_first]),
        "mean_trust_final": _mean([r["final_trust"] for r in with_final]),
        "rewrite_rate": share(normal, lambda r: r["rewrites"] > 0),
        "avg_rewrites": _mean([r["rewrites"] for r in normal]),
        "lie_detected_rate": share(plant, lambda r: r.get("lie_detected")),
        "lie_removed_rate": share(plant, lambda r: r.get("lie_removed")),
        "avg_latency_s": _mean([r["latency_s"] for r in records]),
        "p95_latency_s": p95,
    }


def _pct(v):
    return "n/a" if v is None else f"{v:.0%}"


def _pct1(v):
    return "n/a" if v is None else f"{v:.1%}"


def _num(v, digits=2):
    return "n/a" if v is None else f"{v:.{digits}f}"


def render_markdown(s: dict, meta: dict) -> str:
    rows = [
        ("Final answer correct / partial / incorrect (answerable questions)",
         f"{_pct(s['correct_rate'])} / {_pct(s['partial_rate'])} / {_pct(s['incorrect_rate'])} (n={s['answerable_n']})"),
        ("Correct before the verify loop (first draft)", _pct(s["correct_rate_first_draft"])),
        ("False refusals on answerable questions", _pct(s["false_refusal_rate"])),
        ("Correct refusals on unanswerable questions", f"{_pct(s['refusal_accuracy'])} (n={s['unanswerable_n']})"),
        ("Golden page found in retrieved passages", _pct(s["retrieval_hit_rate"])),
        ("Sentences flagged by the verifier, first draft -> final",
         f"{_pct1(s['first_flagged_rate'])} -> {_pct1(s['final_flagged_rate'])}"),
        ("Answers with at least one flagged sentence, first draft -> final",
         f"{_pct(s['first_answers_with_flags'])} -> {_pct(s['final_answers_with_flags'])}"),
        ("Mean trust score, first draft -> final",
         f"{_num(s['mean_trust_first'])} -> {_num(s['mean_trust_final'])}"),
        ("Answers that needed a rewrite",
         f"{_pct(s['rewrite_rate'])} (average {_num(s['avg_rewrites'])} rewrites per answer)"),
        ("Planted lie detected / removed from the final answer",
         f"{_pct(s['lie_detected_rate'])} / {_pct(s['lie_removed_rate'])} (n={s['plant_n']})"),
        ("Latency per question, average / p95", f"{_num(s['avg_latency_s'], 1)}s / {_num(s['p95_latency_s'], 1)}s"),
    ]
    head = (
        f"Run `{meta['run_name']}` | retrieval `{meta['mode']}` | k={meta['k']} | "
        f"LLM `{meta['llm']}` | {meta['date']}"
    )
    body = "\n".join(f"| {name} | {value} |" for name, value in rows)
    return f"{head}\n\n| Metric | Result |\n|---|---|\n{body}\n"