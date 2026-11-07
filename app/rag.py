from app.llm import generate
from app.retrieval import retrieve

RULES = """You answer questions using ONLY the numbered context passages below.
Rules:
- Use only facts stated in the passages. Do not use outside knowledge.
- After each claim, cite the passage number in square brackets, like [1] or [2][3].
- If the passages do not contain the answer, reply exactly: I could not find this in the document.
- Be concise: 2 to 5 sentences unless the question needs a list.
- Write citations with plain square brackets like [1], never any other bracket style."""


def build_prompt(question: str, chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[{i}] (page {c['page']})\n{c['content']}" for i, c in enumerate(chunks, 1)
    )
    return f"{RULES}\n\nCONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:"


def answer(question: str, k: int = 5, filename: str | None = None) -> dict:
    chunks = retrieve(question, k, filename)
    if not chunks:
        return {"answer": "No documents are ingested yet.", "sources": []}
    text = generate(build_prompt(question, chunks))
    return {"answer": text.strip(), "sources": chunks}