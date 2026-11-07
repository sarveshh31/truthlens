from app.db import get_conn
from app.embeddings import embed_query


def retrieve(question: str, k: int = 5, filename: str | None = None) -> list[dict]:
    """Return the k chunks closest in meaning to the question."""
    q = embed_query(question)
    where = "WHERE d.filename = %s" if filename else ""
    sql = f"""
        SELECT c.id, c.page, c.content, d.filename,
               1 - (c.embedding <=> %s) AS score
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        {where}
        ORDER BY c.embedding <=> %s
        LIMIT %s
    """
    params = [q] + ([filename] if filename else []) + [q, k]
    with get_conn() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [
        {"id": r[0], "page": r[1], "content": r[2], "filename": r[3], "score": float(r[4])}
        for r in rows
    ]