import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import psycopg

from app.db import get_conn, init_db

print("Creating extension, tables and indexes...")
init_db()
print("OK")

conn = get_conn()

version = conn.execute(
    "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
).fetchone()[0]
print("pgvector version:", version)

tables = conn.execute(
    "SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY 1"
).fetchall()
print("Tables:", [t[0] for t in tables])

print("\nSmoke test: insert 2 fake chunks, search, then roll back...")
v1 = np.zeros(384, dtype=np.float32)
v1[0] = 1.0
v2 = np.zeros(384, dtype=np.float32)
v2[1] = 1.0

with conn.transaction():
    doc_id = conn.execute(
        "INSERT INTO documents (filename, page_count) VALUES ('smoke.pdf', 1) RETURNING id"
    ).fetchone()[0]
    conn.execute(
        "INSERT INTO chunks (document_id, chunk_index, page, content, embedding) "
        "VALUES (%s, 0, 1, 'chunk A', %s)",
        (doc_id, v1),
    )
    conn.execute(
        "INSERT INTO chunks (document_id, chunk_index, page, content, embedding) "
        "VALUES (%s, 1, 1, 'chunk B', %s)",
        (doc_id, v2),
    )
    rows = conn.execute(
        "SELECT content, embedding <=> %s AS distance FROM chunks ORDER BY distance LIMIT 2",
        (v1,),
    ).fetchall()
    for content, dist in rows:
        print(f"  {content}: distance {dist:.2f}")
    raise psycopg.Rollback  # undo the fake data

count = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
print("Chunks left in table (should be 0):", count)
print("\nPHASE 2 CHECK PASSED")