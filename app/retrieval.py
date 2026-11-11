import math
import re

from rank_bm25 import BM25Okapi

from app import config
from app.db import get_conn
from app.embeddings import embed_query
from app.rerank import rerank

MODES = ("vector", "hybrid", "hybrid_rerank")
RRF_K = 60  # standard constant for Reciprocal Rank Fusion

_bm25_cache: dict = {}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _where(filename):
    return ("WHERE d.filename = %s", [filename]) if filename else ("", [])


def vector_search(conn, query_vec, n: int, filename=None) -> list[dict]:
    where, params = _where(filename)
    sql = f"""
        SELECT c.id, c.page, c.content, d.filename, 1 - (c.embedding <=> %s) AS score
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        {where}
        ORDER BY c.embedding <=> %s
        LIMIT %s
    """
    rows = conn.execute(sql, [query_vec] + params + [query_vec, n]).fetchall()
    return [
        {"id": r[0], "page": r[1], "content": r[2], "filename": r[3], "score": float(r[4])}
        for r in rows
    ]


def _bm25_index(conn, filename=None) -> dict:
    """Build the BM25 index in memory. Rebuilt automatically when the chunks change."""
    where, params = _where(filename)
    base = "FROM chunks c JOIN documents d ON d.id = c.document_id " + where
    count, max_id = conn.execute(f"SELECT count(*), max(c.id) {base}", params).fetchone()
    key = (count, max_id)
    cached = _bm25_cache.get(filename)
    if cached and cached["key"] == key:
        return cached
    rows = conn.execute(
        f"SELECT c.id, c.page, c.content, d.filename {base} ORDER BY c.id", params
    ).fetchall()
    docs = [{"id": r[0], "page": r[1], "content": r[2], "filename": r[3]} for r in rows]
    bm25 = BM25Okapi([_tokens(d["content"]) for d in docs]) if docs else None
    cached = {"key": key, "docs": docs, "bm25": bm25}
    _bm25_cache[filename] = cached
    return cached


def bm25_search(conn, question: str, n: int, filename=None) -> list[dict]:
    index = _bm25_index(conn, filename)
    if index["bm25"] is None:
        return []
    scores = index["bm25"].get_scores(_tokens(question))
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
    return [dict(index["docs"][i], score=float(scores[i])) for i in order if scores[i] > 0]


def rrf_merge(*ranked_lists: list[dict], k: int = RRF_K) -> list[dict]:
    """Reciprocal Rank Fusion: score = sum of 1 / (k + rank) over every list a chunk appears in."""
    scores: dict = {}
    items: dict = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, 1):
            scores[item["id"]] = scores.get(item["id"], 0.0) + 1.0 / (k + rank)
            items.setdefault(item["id"], item)
    merged = sorted(items.values(), key=lambda it: scores[it["id"]], reverse=True)
    return [dict(it, score=scores[it["id"]]) for it in merged]


def retrieve(question: str, k: int = 5, filename: str | None = None, mode: str | None = None) -> list[dict]:
    """Return the k best chunks. mode: vector | hybrid | hybrid_rerank (default from config)."""
    mode = mode or config.RETRIEVAL_MODE
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    n = max(config.CANDIDATES, k)
    query_vec = embed_query(question)

    with get_conn() as conn:
        if mode == "vector":
            return vector_search(conn, query_vec, k, filename)
        vec = vector_search(conn, query_vec, n, filename)
        kw = bm25_search(conn, question, n, filename)

    merged = rrf_merge(vec, kw)
    if mode == "hybrid":
        return merged[:k]

    top = rerank(question, merged[:n], k)
    for item in top:
        item["score"] = 1.0 / (1.0 + math.exp(-item["rerank_score"]))  # squash to 0..1
    return top



def merge_page_group(chunks, page_chunks, max_extra=6):
    """Put the whole page of the top hit in reading order, then the other hits.

    chunks: ranked retrieval results. page_chunks: every chunk on the top hit's page, ordered by chunk_index.
    Chunks added this way carry expanded=True and score 0.0.
    """
    if not chunks or not page_chunks:
        return chunks
    by_id = {c["id"]: c for c in chunks}
    group, extra = [], 0
    for pc in page_chunks:
        if pc["id"] in by_id:
            group.append(by_id[pc["id"]])
        elif extra < max_extra:
            group.append(dict(pc, score=0.0, expanded=True))
            extra += 1
    in_group = {c["id"] for c in group}
    return group + [c for c in chunks if c["id"] not in in_group]


def expand_page(chunks, max_extra=6):
    """Add the other chunks from the top hit's page, so lists that span several chunks come back whole."""
    if not chunks:
        return chunks
    top = chunks[0]
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT c.id, c.page, c.content, d.filename
            FROM chunks c JOIN documents d ON d.id = c.document_id
            WHERE c.document_id = (SELECT document_id FROM chunks WHERE id = %s) AND c.page = %s
            ORDER BY c.chunk_index
            """,
            [top["id"], top["page"]],
        ).fetchall()
    page_chunks = [{"id": r[0], "page": r[1], "content": r[2], "filename": r[3]} for r in rows]
    return merge_page_group(chunks, page_chunks, max_extra)