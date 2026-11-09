import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from app.evalutils import first_relevant_rank, load_golden, summarize
from app.retrieval import MODES, retrieve

parser = argparse.ArgumentParser(description="Retrieval evaluation (makes no LLM calls)")
parser.add_argument("--golden", default="eval/golden.json")
parser.add_argument("--k", type=int, default=5)
parser.add_argument("--filename", default=None, help="restrict to one ingested PDF")
parser.add_argument("--out", default="eval/results/retrieval_latest.json")
args = parser.parse_args()

try:
    items, problems = load_golden(args.golden)
except (OSError, json.JSONDecodeError) as e:
    sys.exit(f"Cannot read {args.golden}: {e}")
if problems:
    print("Fix these problems in the golden set first:")
    for p in problems:
        print("  -", p)
    sys.exit(1)

answerable = [it for it in items if it["answerable"]]
print(f"Golden set: {len(items)} questions, {len(answerable)} answerable (used for retrieval)")
if not answerable:
    sys.exit("No answerable questions found.")

retrieve("warm up", args.k, args.filename, mode="hybrid_rerank")  # load models and the BM25 index once

results, misses = {}, {}
for mode in MODES:
    ranks, latencies, missed = [], [], []
    for it in answerable:
        start = time.time()
        chunks = retrieve(it["question"], args.k, args.filename, mode=mode)
        latencies.append(time.time() - start)
        rank = first_relevant_rank(chunks, it)
        ranks.append(rank)
        if rank is None:
            missed.append(f"#{it['id']} {it['question']}")
    summary = summarize(ranks, args.k)
    summary["avg_latency_s"] = round(sum(latencies) / len(latencies), 3)
    results[mode] = summary
    misses[mode] = missed
    print(f"  {mode}: done")

hit_k = f"hit@{args.k}"
print(f"\n| Mode | hit@1 | hit@3 | {hit_k} | MRR | avg latency |")
print("|---|---|---|---|---|---|")
for mode, s in results.items():
    print(
        f"| {mode} | {s['hit@1']:.0%} | {s['hit@3']:.0%} | {s[hit_k]:.0%} "
        f"| {s['mrr']:.3f} | {s['avg_latency_s']:.2f}s |"
    )

print(f"\nQuestions with NO relevant chunk in the top {args.k}:")
for mode, missed in misses.items():
    print(f"  {mode}: {len(missed)}")
    for line in missed:
        print(f"      {line}")

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(
    json.dumps(
        {
            "generated": datetime.now().isoformat(timespec="seconds"),
            "k": args.k,
            "answerable_questions": len(answerable),
            "results": results,
            "misses": misses,
        },
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)
print(f"\nSaved to {out}")