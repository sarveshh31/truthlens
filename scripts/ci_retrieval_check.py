"""CI regression gate: ingest a tiny fictional PDF and fail if retrieval quality drops.

No LLM and no API keys are used, only the embedding model, BM25, the reranker and PostgreSQL.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from app import config
from app.evalutils import first_relevant_rank, load_golden
from app.db import init_db
from app.ingest import ingest_pdf
from app.retrieval import MODES, retrieve

parser = argparse.ArgumentParser()
parser.add_argument("--pdf", default="data/vantora_demo.pdf")
parser.add_argument("--golden", default="eval/ci_golden.json")
parser.add_argument("--mode", choices=MODES, default="hybrid_rerank")
parser.add_argument("--k", type=int, default=5)
parser.add_argument("--min-hit", type=float, default=0.85, help="minimum hit@k")
parser.add_argument("--min-mrr", type=float, default=0.55, help="minimum MRR")
args = parser.parse_args()

config.RETRIEVAL_MODE = args.mode
items, problems = load_golden(args.golden)
if problems:
    sys.exit("Golden set problems: " + "; ".join(problems))

init_db()
ingest_pdf(args.pdf)
name = Path(args.pdf).name

ranks = []
for it in items:
    chunks = retrieve(it["question"], args.k, name, args.mode)
    rank = first_relevant_rank(chunks, it)
    ranks.append(rank)
    print(f"  #{it['id']:>2} rank={rank} | {it['question'][:60]}")

n = len(ranks)
hit1 = sum(1 for r in ranks if r == 1) / n
hit = sum(1 for r in ranks if r is not None and r <= args.k) / n
mrr = sum(1 / r for r in ranks if r is not None) / n
print(f"\nmode={args.mode} n={n} hit@1={hit1:.2f} hit@{args.k}={hit:.2f} MRR={mrr:.2f}")
print(f"thresholds: hit@{args.k} >= {args.min_hit}, MRR >= {args.min_mrr}")
if hit < args.min_hit or mrr < args.min_mrr:
    sys.exit("RETRIEVAL REGRESSION: quality is below the threshold.")
print("RETRIEVAL CHECK PASSED")