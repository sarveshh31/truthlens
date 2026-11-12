import logging
import threading
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app import config
from app.db import get_conn
from app.embeddings import get_model
from app.graph import get_graph, run

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("truthlens")

_stats = {"requests": 0, "errors": 0, "total_ms": 0, "rewrites": 0, "trust_sum": 0.0}
_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading embedding model and building the agent graph...")
    get_model()
    get_graph()
    log.info("TruthLens API ready")
    yield


app = FastAPI(
    title="TruthLens",
    version="0.1.0",
    description="Self-verifying agentic RAG: answers from your documents and checks every sentence.",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    k: int = Field(default=5, ge=1, le=10, description="how many passages to retrieve")
    filename: Optional[str] = Field(default=None, description="restrict to one ingested PDF")
    debug_plant_lie: bool = Field(default=False, description="testing only: inject a fake sentence")


class SentenceOut(BaseModel):
    index: int
    text: str
    verdict: str  # supported | partial | unsupported
    passage: Optional[int] = None
    reason: str = ""


class SourceOut(BaseModel):
    id: int
    page: Optional[int] = None
    filename: str
    score: float
    content: str
    expanded: bool = False


class AskResponse(BaseModel):
    question: str
    answer: str
    trust: float
    first_draft_trust: float
    refused: bool
    rewrites: int
    sentences: list[SentenceOut]
    first_answer: str = ""
    first_sentences: list[SentenceOut] = []
    sources: list[SourceOut]
    latency_ms: int


@app.get("/health")
def health():
    try:
        with get_conn() as conn:
            conn.execute("SELECT 1")
        db = "ok"
    except Exception as e:
        db = f"error: {type(e).__name__}"
    return {
        "status": "ok" if db == "ok" else "degraded",
        "database": db,
        "llm_provider": config.LLM_PROVIDER,
    }


@app.get("/documents")
def documents():
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT d.id, d.filename, d.page_count, count(c.id)
            FROM documents d LEFT JOIN chunks c ON c.document_id = d.id
            GROUP BY d.id ORDER BY d.id
            """
        ).fetchall()
    return [{"id": r[0], "filename": r[1], "pages": r[2], "chunks": r[3]} for r in rows]


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    start = time.time()
    try:
        state = run(req.question, req.k, req.filename, req.debug_plant_lie)
    except (Exception, SystemExit) as e:
        log.exception("ask failed")
        with _lock:
            _stats["errors"] += 1
        raise HTTPException(
            status_code=502, detail=f"The AI pipeline failed ({type(e).__name__}). See server log."
        ) from e

    latency_ms = int((time.time() - start) * 1000)
    v = state["verification"]
    with _lock:
        _stats["requests"] += 1
        _stats["total_ms"] += latency_ms
        _stats["rewrites"] += state["rewrites"]
        _stats["trust_sum"] += v["trust"]
    log.info(
        "ask | %d ms | trust %.0f%% | rewrites %d | %s",
        latency_ms, v["trust"] * 100, state["rewrites"], req.question[:60],
    )
    return AskResponse(
        question=req.question,
        answer=state["answer"],
        trust=v["trust"],
        first_draft_trust=state["first_verification"]["trust"],
        refused=v["refused"],
        rewrites=state["rewrites"],
        sentences=[SentenceOut(**s) for s in v["sentences"]],
        first_answer=state.get("first_answer", ""),
        first_sentences=[SentenceOut(**s) for s in state["first_verification"]["sentences"]],
        sources=[SourceOut(**c) for c in state["chunks"]],
        latency_ms=latency_ms,
    )


@app.get("/metrics")
def metrics():
    with _lock:
        n = _stats["requests"]
        return {
            "requests": n,
            "errors": _stats["errors"],
            "avg_latency_ms": round(_stats["total_ms"] / n) if n else 0,
            "avg_rewrites": round(_stats["rewrites"] / n, 2) if n else 0,
            "avg_trust": round(_stats["trust_sum"] / n, 3) if n else 0,
        }