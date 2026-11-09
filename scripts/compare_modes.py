import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from app.retrieval import MODES, retrieve

if len(sys.argv) < 2:
    sys.exit('Usage: python scripts/compare_modes.py "your question"')
question = sys.argv[1]
print("Question:", question)

for mode in MODES:
    start = time.time()
    results = retrieve(question, 5, mode=mode)
    print(f"\n--- {mode} ({time.time() - start:.2f}s) ---")
    for i, c in enumerate(results, 1):
        preview = c["content"][:90].replace("\n", " ")
        print(f"  [{i}] page {c['page']:>3} | score {c['score']:.3f} | {preview}...")