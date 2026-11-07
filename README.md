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

## Run the API
```bash
uvicorn app.api:app --port 8000
# then open http://127.0.0.1:8000/docs
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