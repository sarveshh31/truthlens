import re
from typing import Literal, Optional

from pydantic import BaseModel, Field

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
Judge ONLY against the passages, never against your own knowledge.
Ignore citation markers like [1] inside the sentences.
Return exactly one verdict per sentence."""


def split_sentences(text: str) -> list[str]:
    """Simple sentence splitter. Also handles bullet lists."""
    sentences = []
    for line in text.strip().split("\n"):
        line = line.strip(" -*\u2022\t")
        if not line:
            continue
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\[\"'(])", line)
        sentences.extend(p.strip() for p in parts if p.strip())
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
    prompt = f"{JUDGE_RULES}\n\nPASSAGES:\n{passages}\n\nSENTENCES TO CHECK:\n{numbered}"

    judge = get_chat_model(0.0).with_structured_output(VerificationResult)
    result = judge.invoke(prompt)
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