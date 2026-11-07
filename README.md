# TruthLens

A self-verifying agentic RAG system: it answers questions from your PDFs, checks every sentence against the sources, and rewrites unsupported claims.

**Status:** under active development. Day 1 done: ingestion + hybrid-ready vector search + basic RAG answers.

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