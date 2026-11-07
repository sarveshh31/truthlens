FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN useradd --create-home --uid 1000 app
WORKDIR /app

# CPU-only PyTorch first: much smaller than the default GPU build
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app

USER app
ENV HF_HOME=/home/app/.cache/huggingface

# download the embedding model at build time so the container starts fast and works offline
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"
ENV HF_HUB_OFFLINE=1

EXPOSE 8000
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]