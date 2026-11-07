import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from app.rag import answer


def show(question: str, k: int) -> None:
    start = time.time()
    result = answer(question, k)
    print("\nANSWER:\n" + result["answer"])
    print("\nSOURCES:")
    for i, c in enumerate(result["sources"], 1):
        preview = c["content"][:110].replace("\n", " ")
        print(f"  [{i}] page {c['page']} | score {c['score']:.2f} | {preview}...")
    print(f"\n({time.time() - start:.1f}s)")


parser = argparse.ArgumentParser(description="Ask TruthLens a question")
parser.add_argument("question", nargs="?", help="leave empty for interactive mode")
parser.add_argument("--k", type=int, default=5, help="how many chunks to retrieve")
args = parser.parse_args()

if args.question:
    show(args.question, args.k)
else:
    while True:
        q = input("\nAsk (blank to quit): ").strip()
        if not q:
            break
        show(q, args.k)