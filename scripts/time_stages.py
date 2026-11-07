import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def t(label, fn):
    start = time.time()
    result = fn()
    print(f"{label}: {time.time() - start:.1f}s", flush=True)
    return result


t("import torch", lambda: __import__("torch"))
t("import sentence_transformers", lambda: __import__("sentence_transformers"))

from app.embeddings import embed_query, get_model

t("load embedding model", get_model)
t("embed one query", lambda: embed_query("What was the total revenue?"))

from app.db import get_conn

conn = t("connect to database", get_conn)
t("run SELECT 1", lambda: conn.execute("SELECT 1").fetchone())

from app.retrieval import retrieve

t("full vector search", lambda: retrieve("What was the total revenue?", 5))
print("DONE")