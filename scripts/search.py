import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from app.retrieval import retrieve

parser = argparse.ArgumentParser()
parser.add_argument("question")
parser.add_argument("--k", type=int, default=5)
args = parser.parse_args()

for i, c in enumerate(retrieve(args.question, args.k), 1):
    preview = c["content"][:300].replace("\n", " ")
    print(f"\n[{i}] page {c['page']} | score {c['score']:.2f}\n    {preview}...")