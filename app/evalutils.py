import json
import re
from pathlib import Path


def normalize(text: str) -> str:
    """Lowercase and drop everything except letters and digits, so '22,465' matches '22 465'."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def is_relevant(chunk: dict, item: dict) -> bool:
    """A chunk is relevant if it is on an expected page or contains an expected keyword."""
    if chunk.get("page") in set(item.get("pages", [])):
        return True
    content = normalize(chunk["content"])
    for keyword in item.get("keywords", []):
        key = normalize(keyword)
        if key and key in content:
            return True
    return False


def first_relevant_rank(chunks: list[dict], item: dict):
    """1-based rank of the first relevant chunk, or None if there is none."""
    for rank, chunk in enumerate(chunks, 1):
        if is_relevant(chunk, item):
            return rank
    return None


def summarize(ranks: list, k: int) -> dict:
    n = len(ranks)

    def hit(m: int) -> float:
        return sum(1 for r in ranks if r is not None and r <= m) / n if n else 0.0

    mrr = sum(1.0 / r for r in ranks if r is not None) / n if n else 0.0
    return {"n": n, "hit@1": hit(1), "hit@3": hit(3), f"hit@{k}": hit(k), "mrr": mrr}


def load_golden(path):
    """Read the golden set and return (items, problems). Problems is a list of readable messages."""
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    problems, seen = [], set()
    for pos, it in enumerate(items, 1):
        tag = f"item {pos}"
        if not isinstance(it.get("id"), int):
            problems.append(f"{tag}: 'id' must be a number")
        elif it["id"] in seen:
            problems.append(f"{tag}: duplicate id {it['id']}")
        else:
            seen.add(it["id"])
        if not str(it.get("question", "")).strip():
            problems.append(f"{tag}: empty question")
        if not isinstance(it.get("answerable"), bool):
            problems.append(f"{tag}: 'answerable' must be true or false")
        elif it["answerable"]:
            pages = it.get("pages", [])
            if not pages or not all(isinstance(p, int) for p in pages):
                problems.append(f"{tag}: answerable questions need 'pages' as a list of numbers")
            for kw in it.get("keywords", []):
                if len(normalize(kw)) < 3:
                    problems.append(f"{tag}: keyword '{kw}' is too short")
    return items, problems