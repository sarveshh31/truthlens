# TruthLens

A self-verifying agentic RAG system: it answers questions from your PDFs, checks every sentence against the sources, and rewrites unsupported claims.

## Status
Working: PDF ingestion, local embeddings, pgvector retrieval, a LangGraph agent that generates, verifies every sentence and rewrites unsupported claims, and a FastAPI service. Coming: hybrid search and reranking, evaluation pipeline with MLflow, CI, Docker image, Streamlit UI.

## Architecture

```mermaid
flowchart LR
    PDF["PDF documents"] --> ING["Ingestion<br/>pypdf and chunking"]
    ING --> EMB["Embeddings<br/>bge-small, local"]
    EMB --> DB[("PostgreSQL + pgvector")]
    USER["User"] --> API["FastAPI<br/>/ask /health /documents /metrics"]
    API --> AGENT["LangGraph agent"]
    AGENT --> DB
    AGENT <--> LLM["LLM via LangChain<br/>Groq or Gemini"]
```

## The agent loop

```mermaid
flowchart TD
    Q["Question"] --> R["Retrieve top passages<br/>pgvector similarity"]
    R --> G["Generate answer<br/>from passages only"]
    G --> V["Verify each sentence<br/>LLM judge"]
    V -->|all supported| OUT["Final answer with trust score"]
    V -->|flagged and rewrites left| W["Rewrite flagged sentences"]
    W --> V
    V -->|rewrites used up| OUT
```

## Run with Docker (recommended)
```bash
cp .env.example .env        # then add your Groq (or Gemini) key and model name
docker compose up -d --build
# API docs: http://127.0.0.1:8000/docs
```

## Ingest a PDF
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/init_db.py
python scripts/ingest_pdf.py data/your_file.pdf
```
The PDF is not included in this repo. Download one yourself and put it in `data/`.

## Tests
```bash
python -m pytest -q
```

## Stack
Python, PostgreSQL + pgvector, sentence-transformers (bge-small), Google Gemini API (free tier), Docker Compose. Coming: LangGraph, FastAPI, MLflow, GitHub Actions.

## Run locally
```bash
git clone <this repo>
cd truthlens
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env                                 # add your Gemini API key and model name
docker compose up -d
python scripts/init_db.py
python scripts/ingest_pdf.py data/your_file.pdf
python scripts/ask.py "your question"
```