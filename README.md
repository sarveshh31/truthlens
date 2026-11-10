# TruthLens

![CI](https://github.com/sarveshh31/truthlens/actions/workflows/ci.yml/badge.svg)

**A self-verifying RAG agent.** It answers questions about a PDF with citations, then fact-checks every sentence of its own answer against the source passages. Anything unsupported is rewritten before the user sees it, and every answer comes with a trust score.

**Stack:** LangGraph · LangChain · FastAPI · PostgreSQL + pgvector · BM25 + cross-encoder reranking · Docker Compose · MLflow · GitHub Actions · pytest

## Why this exists
Plain RAG can still state things its sources do not say. TruthLens adds a verification loop: each sentence is judged `supported`, `partial` or `unsupported` against the retrieved passages, and flagged sentences trigger a rewrite (at most 2 rounds). The trust score is the share of supported sentences, so the user can see how much to rely on an answer.

## Architecture

```mermaid
flowchart LR
    Q[Question] --> R[Retrieve<br/>vector + BM25 + RRF + rerank]
    R --> G[Generate answer<br/>with citations]
    G --> V[Verify each sentence<br/>LLM judge]
    V -->|all supported| A[Answer + trust score]
    V -->|flagged sentences| W[Rewrite]
    W --> V
    DB[(PostgreSQL<br/>pgvector)] --- R
```

- **Retrieval:** `BAAI/bge-small-en-v1.5` embeddings in pgvector (HNSW, cosine) plus BM25, merged with Reciprocal Rank Fusion (k=60), then reranked with `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- **Agent:** a LangGraph state machine: retrieve → generate → verify → (rewrite → verify) up to 2 times.
- **LLM:** provider-switchable (Groq `openai/gpt-oss-120b` or Gemini) behind one `get_chat_model()` function, so the free tier of either can be used.
- **API:** FastAPI with `/health`, `/documents`, `/ask` and `/metrics`, packaged with Docker Compose (API + PostgreSQL/pgvector).

## Results

Evaluated on a hand-built golden set of **30 questions** about the Capgemini 2025 Integrated Annual Report (26 answerable with reference answers and source pages, 4 unanswerable). Every reference answer was checked against the PDF text.

### Retrieval (26 answerable questions, k=5)

| Mode | hit@1 | hit@3 | hit@5 | MRR | Avg latency |
|---|---|---|---|---|---|
| Vector only | 88% | 100% | 100% | 0.942 | 0.07 s |
| Hybrid (vector + BM25 + RRF) | 85% | 100% | 100% | 0.917 | 0.08 s |
| Hybrid + cross-encoder rerank (default) | 88% | 100% | 100% | 0.942 | 1.30 s |

A retrieved chunk counts as a hit if it is on an expected page or contains an expected keyword. **All three modes found a relevant chunk for every question within the top 5**, so this golden set cannot separate them on recall. Hybrid and rerank did not beat plain vector search here; their advantage would need a harder set with more keyword-heavy or ambiguous questions.

### End-to-end (hybrid + rerank, Groq `openai/gpt-oss-120b`)

| Metric | Result |
|---|---|
| Final answer correct (LLM-graded vs reference) | **25 of 26** |
| Correct refusals on unanswerable questions | **4 of 4** |
| False refusals on answerable questions | 1 of 26 |
| Golden page present in retrieved passages | 26 of 26 |
| Planted false sentence detected by the verifier | **10 of 10** |
| Planted false sentence removed from the final answer | **10 of 10** |
| Sentences flagged by the verifier, first draft → final | 3.9% → 0.0% |
| Answers with at least one flagged sentence, first draft → final | 4% → 0% |
| Mean trust score, first draft → final | 0.96 → 1.00 |
| Answers that needed a rewrite | 3% (0.03 rewrites per answer on average) |
| Latency per question in the batch evaluation, average / p95 | 29.0 s / 54.1 s |

**Planted-lie test:** for 10 questions, a false sentence ("The company was founded on Mars in 1850 and employs 3 million people.") is injected into the draft answer before verification. This checks that the verifier catches fabrications and that the rewrite loop removes them.

All runs are tracked in MLflow (`mlflow ui --backend-store-uri sqlite:///mlflow.db`).

## Limitations (read these before trusting the numbers)
- **Small golden set.** 30 questions on one document: results are indicative, not statistically tight. One question is about 4 points.
- **Retrieval is saturated.** Every mode reaches 100% hit@5, so the retrieval comparison shows no winner on recall. The differences in hit@1 and MRR are within the noise of 26 questions.
- **Same-family judge.** Correctness is graded by an LLM from the same family as the answerer. The planted-lie test does not depend on that judge.
- **The verification loop barely changed accuracy here.** Correctness was 96% before and after it, because the first drafts were already mostly faithful. Its measured value is catching injected falsehoods, not raising accuracy on this set.
- **The flagged-sentence drop (3.9% → 0%) rests on very few sentences.** The verifier also drives the rewrite, so this metric is partly circular.
- **One false refusal.** On "Which five technology trends does the Group highlight…", the system refused although the right page was retrieved. List questions that spread over several passages are the weak spot.
- **Latency.** The batch average of ~29 s includes the verification loop and free-tier rate-limit pauses. A single interactive request takes about 3 s.
- **Scope.** English, text-based PDFs only. Scanned PDFs are not supported.

## Run it locally

```bash
git clone https://github.com/sarveshh31/truthlens.git
cd truthlens
cp .env.example .env        # add your GROQ_API_KEY (free tier)
docker compose up -d --build
```

Open http://127.0.0.1:8000/docs, then ingest a PDF and ask questions:

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/init_db.py
python scripts/ingest_pdf.py data/your_report.pdf
python scripts/ask.py "Who is the CEO?"
```

Example response from `POST /ask`:

```json
{
  "question": "Who is the CEO?",
  "answer": "The Chief Executive Officer is Aiman Ezzat [1][3].",
  "trust": 1,
  "refused": false,
  "rewrites": 0,
  "sentences": [{"text": "The Chief Executive Officer is Aiman Ezzat [1][3].", "verdict": "supported", "passage": 1}],
  "sources": [{"page": 4, "filename": "capgemini_report.pdf", "content": "…"}]
}
```

## Evaluate

```bash
python scripts/eval_retrieval.py     # retrieval metrics for all three modes
python scripts/eval_agent.py         # end-to-end metrics + MLflow logging
pytest -q                            # unit tests
```

## CI
Every push and pull request runs on GitHub Actions: unit tests, then a **retrieval regression check** that builds a small fictional PDF, ingests it into a pgvector service container and fails the build if hit@5 or MRR drops below a threshold (hit@5 ≥ 0.85, MRR ≥ 0.55). The demo corpus has only 6 chunks, so hit@5 is easy there and MRR is the meaningful signal (0.96 on the latest local run). CI uses no LLM calls and no secrets.

## Project layout

```
app/        config, db, ingest, embeddings, retrieval, rerank, chat, verify, graph, api, evaluation helpers
scripts/    CLI tools: ingest, ask, evaluation, CI check, demo PDF builder
eval/       golden set (golden.json) and CI golden set (ci_golden.json)
tests/      pytest unit tests
```

## Author
**Sarvesh Tiwari**, final-year B.Tech Information Technology, RCOEM Nagpur · [github.com/sarveshh31](https://github.com/sarveshh31)