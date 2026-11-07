import re
from typing import Literal, Optional

from pydantic import BaseModel, Field

from app import config

from app.chat import get_chat_model

NOT_FOUND = "I could not find this in the document."


class SentenceVerdict(BaseModel):
    index: int = Field(description="1-based number of the sentence")
    verdict: Literal["supported", "partial", "unsupported"]
    passage: Optional[int] = Field(
        default=None, description="number of the passage that supports it, or null"
    )
    reason: str = Field(default="", description="one short reason")


class VerificationResult(BaseModel):
    verdicts: list[SentenceVerdict]


JUDGE_RULES = """You are a strict fact-checker.
For each numbered sentence, decide whether the PASSAGES support it.
- supported: the passages clearly state this (numbers, names and dates must match).
- partial: the passages support only part of the sentence, or one detail is not confirmed.
- unsupported: the passages do not state this, or they contradict it.
- Every specific detail in the sentence (place, country, date, number, name, scope) must appear in the passages. If the sentence adds a detail the passages do not mention, mark it partial.
Judge ONLY against the passages, never against your own knowledge.
Ignore citation markers like [1] inside the sentences.
Return exactly one verdict per sentence."""

JSON_FORMAT = """Return ONLY a JSON object in exactly this shape, with one entry per sentence:
{"verdicts": [{"index": 1, "verdict": "supported", "passage": 2, "reason": "short reason"}]}
"verdict" must be one of: supported, partial, unsupported.
"passage" is the supporting passage number, or null."""


def run_judge(prompt: str) -> VerificationResult:
    """Ask the LLM for verdicts as JSON. Retries up to 3 times if the model answers badly."""
    # Groq: JSON mode is more reliable than tool calling. Gemini: keep the default method.
    kwargs = {"method": "json_mode"} if config.LLM_PROVIDER == "groq" else {}
    judge = get_chat_model(0.0).with_structured_output(VerificationResult, **kwargs)
    last_error = None
    for _ in range(3):
        try:
            return judge.invoke(prompt)
        except Exception as e:  # bad JSON, tool error, etc.
            last_error = e
    raise last_error


LIST_PREFIX = re.compile(r"^\s*(?:[-*\u2022]+\s*)?(?:\d{1,2}[.)]\s*)?")


def split_sentences(text: str) -> list[str]:
    """Split an answer into checkable sentences. Handles numbered lists, bullets and headers."""
    sentences = []
    for line in text.strip().split("\n"):
        line = line.replace("**", "").strip()
        line = LIST_PREFIX.sub("", line).strip()
        if not line or line.endswith(":"):  # empty line or a lead-in like "The priorities are:"
            continue
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\[\"'(\u3010])", line)
        for p in parts:
            p = p.strip()
            if re.search(r"[A-Za-z]{3}", p):  # skip fragments with no real words
                sentences.append(p)
    return sentences

def verify_answer(answer: str, chunks: list[dict]) -> dict:
    """Check every sentence of the answer against the retrieved chunks."""
    if answer.strip() == NOT_FOUND:
        return {"sentences": [], "trust": 1.0, "all_supported": True, "refused": True}

    sentences = split_sentences(answer)
    if not sentences:
        return {"sentences": [], "trust": 1.0, "all_supported": True, "refused": False}

    passages = "\n\n".join(
        f"[{i}] (page {c['page']})\n{c['content']}" for i, c in enumerate(chunks, 1)
    )
    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences, 1))
    prompt = (
        f"{JUDGE_RULES}\n\n{JSON_FORMAT}\n\n"
        f"PASSAGES:\n{passages}\n\nSENTENCES TO CHECK:\n{numbered}"
    )
    result = run_judge(prompt)
    by_index = {v.index: v for v in result.verdicts}

    out = []
    for i, s in enumerate(sentences, 1):
        v = by_index.get(i)
        out.append(
            {
                "index": i,
                "text": s,
                "verdict": v.verdict if v else "unsupported",  # no verdict = not trusted
                "passage": v.passage if v else None,
                "reason": v.reason if v else "judge returned no verdict",
            }
        )
    supported = sum(1 for x in out if x["verdict"] == "supported")
    return {
        "sentences": out,
        "trust": supported / len(out),
        "all_supported": supported == len(out),
        "refused": False,
    }