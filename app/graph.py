import os
from typing import Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from app.chat import get_chat_model, message_text
from app.rag import build_prompt
from app.retrieval import expand_page, retrieve
from app.verify import NOT_FOUND, verify_answer

MAX_REWRITES = 2
EXPAND_PAGE = os.getenv("EXPAND_PAGE", "1") != "0"  # set EXPAND_PAGE=0 to switch it off for comparisons
PLANTED_LIE = " The company was founded on Mars in 1850 and employs 3 million people."

REWRITE_RULES = """You fix an answer so that every sentence is backed by the passages.
You receive the QUESTION, the PASSAGES, the current ANSWER and a fact-check report.
- Keep sentences marked supported exactly as they are.
- For partial: keep only the part the passages support, or correct the detail using the passages.
- For unsupported: delete the sentence, unless the passages let you correct it.
- Never add facts that are not in the passages.
- Keep citations like [1] and make sure they point to the right passage.
- If nothing supported remains, reply exactly: I could not find this in the document.
Return only the corrected answer, with no explanation."""


class GraphState(TypedDict, total=False):
    # inputs
    question: str
    k: int
    filename: Optional[str]
    plant_lie: bool          # test switch: inject a fake sentence into the first draft
    # working data
    chunks: list[dict]
    answer: str
    verification: dict
    rewrites: int
    # for demos and evaluation
    first_answer: str
    first_verification: dict
    trace: list[dict]


def _trace(state: GraphState, step: str, detail: str) -> list[dict]:
    return state.get("trace", []) + [{"step": step, "detail": detail}]


def retrieve_node(state: GraphState):
    chunks = retrieve(state["question"], state.get("k", 5), state.get("filename"))
    if EXPAND_PAGE:
        chunks = expand_page(chunks)
    return {
        "chunks": chunks,
        "rewrites": 0,
        "trace": _trace(state, "retrieve", f"{len(chunks)} passages"),
    }


def generate_node(state: GraphState):
    chunks = state["chunks"]
    if not chunks:
        return {"answer": NOT_FOUND, "trace": _trace(state, "generate", "no passages found")}
    resp = get_chat_model(0.2).invoke(build_prompt(state["question"], chunks))
    text = message_text(resp).strip()
    if state.get("plant_lie"):
        text += PLANTED_LIE
    return {"answer": text, "trace": _trace(state, "generate", f"{len(text)} characters")}


def verify_node(state: GraphState):
    v = verify_answer(state["answer"], state["chunks"])
    flagged = sum(1 for s in v["sentences"] if s["verdict"] != "supported")
    update = {
        "verification": v,
        "trace": _trace(state, "verify", f"trust {v['trust']:.0%}, {flagged} sentence(s) flagged"),
    }
    if "first_verification" not in state:  # remember the BEFORE picture
        update["first_answer"] = state["answer"]
        update["first_verification"] = v
    return update


def rewrite_node(state: GraphState):
    v = state["verification"]
    report = "\n".join(
        f"{s['index']}. [{s['verdict']}] {s['text']}  -- {s['reason']}" for s in v["sentences"]
    )
    passages = "\n\n".join(
        f"[{i}] (page {c['page']})\n{c['content']}" for i, c in enumerate(state["chunks"], 1)
    )
    prompt = (
        f"{REWRITE_RULES}\n\nQUESTION: {state['question']}\n\nPASSAGES:\n{passages}\n\n"
        f"CURRENT ANSWER:\n{state['answer']}\n\nFACT-CHECK REPORT:\n{report}\n\nCORRECTED ANSWER:"
    )
    new_answer = message_text(get_chat_model(0.0).invoke(prompt)).strip() or NOT_FOUND
    n = state["rewrites"] + 1
    return {
        "answer": new_answer,
        "rewrites": n,
        "trace": _trace(state, "rewrite", f"attempt {n} of {MAX_REWRITES}"),
    }


def route_after_verify(state: GraphState) -> str:
    if state["verification"]["all_supported"]:
        return "done"
    if state["rewrites"] >= MAX_REWRITES:
        return "done"
    return "rewrite"


def build_graph():
    g = StateGraph(GraphState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("generate", generate_node)
    g.add_node("verify", verify_node)
    g.add_node("rewrite", rewrite_node)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "verify")
    g.add_conditional_edges("verify", route_after_verify, {"rewrite": "rewrite", "done": END})
    g.add_edge("rewrite", "verify")
    return g.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run(question: str, k: int = 5, filename: Optional[str] = None, plant_lie: bool = False) -> dict:
    """Run the whole agent and return the final state."""
    return get_graph().invoke(
        {"question": question, "k": k, "filename": filename, "plant_lie": plant_lie}
    )