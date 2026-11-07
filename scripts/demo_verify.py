import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from app.rag import answer
from app.verify import verify_answer

question = sys.argv[1] if len(sys.argv) > 1 else "What are the main strategic priorities?"
print("Question:", question)

result = answer(question, k=5)
chunks = result["sources"]
print("\nANSWER FROM GEMINI:\n" + result["answer"])

TAGS = {"supported": "[OK]  ", "partial": "[??]  ", "unsupported": "[XX]  "}


def show(title: str, text: str) -> None:
    v = verify_answer(text, chunks)
    print(f"\n--- {title} | trust {v['trust']:.0%} ---")
    for s in v["sentences"]:
        print(f"{TAGS[s['verdict']]}{s['text']}")
        print(f"        passage={s['passage']} | {s['reason']}")


show("REAL ANSWER", result["answer"])

fake = result["answer"] + " The company was founded on Mars in 1850 and employs 3 million people."
show("SAME ANSWER WITH A PLANTED LIE", fake)