import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from app.graph import run

TAGS = {"supported": "[OK]  ", "partial": "[??]  ", "unsupported": "[XX]  "}


def show(title: str, text: str, v: dict) -> None:
    print(f"\n=== {title} | trust {v['trust']:.0%} ===")
    if v["refused"] or not v["sentences"]:
        print(text)
        return
    for s in v["sentences"]:
        print(f"{TAGS[s['verdict']]}{s['text']}")


parser = argparse.ArgumentParser(description="Run the TruthLens agent")
parser.add_argument("question")
parser.add_argument("--k", type=int, default=5)
parser.add_argument("--plant", action="store_true", help="inject a fake sentence to test the loop")
args = parser.parse_args()

start = time.time()
state = run(args.question, args.k, plant_lie=args.plant)
seconds = time.time() - start

print("\nSTEPS:")
for t in state["trace"]:
    print(f"  - {t['step']}: {t['detail']}")

show("BEFORE (first draft)", state["first_answer"], state["first_verification"])
show("AFTER (final answer)", state["answer"], state["verification"])

print(f"\nRewrites used: {state['rewrites']} | time: {seconds:.1f}s")
print("\nSOURCES:")
for i, c in enumerate(state["chunks"], 1):
    print(f"  [{i}] page {c['page']} | score {c['score']:.2f}")