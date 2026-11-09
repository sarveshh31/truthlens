from functools import lru_cache

from sentence_transformers import CrossEncoder

from app import config


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    """Load the cross-encoder once (first use downloads about 90 MB)."""
    return CrossEncoder(config.RERANK_MODEL)


def rerank(question: str, chunks: list[dict], top_k: int) -> list[dict]:
    """Score every (question, passage) pair and keep the best top_k."""
    if not chunks:
        return []
    scores = get_reranker().predict([(question, c["content"]) for c in chunks])
    ranked = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)
    out = []
    for chunk, score in ranked[:top_k]:
        item = dict(chunk)
        item["rerank_score"] = float(score)
        out.append(item)
    return out