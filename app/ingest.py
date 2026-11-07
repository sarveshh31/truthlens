import re
import time
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.db import get_conn
from app.embeddings import embed_texts

CHUNK_SIZE = 1200      # characters per chunk (about 300 tokens)
CHUNK_OVERLAP = 200    # characters shared between neighbouring chunks
MIN_CHARS = 40         # ignore near-empty pages and tiny fragments


def clean_text(text: str) -> str:
    text = text.replace("\x00", "")        # PostgreSQL rejects null bytes
    text = re.sub(r"[ \t]+", " ", text)    # collapse repeated spaces
    text = re.sub(r"\n{3,}", "\n\n", text) # collapse blank lines
    return text.strip()


def extract_pages(pdf_path: Path, max_pages: int | None = None):
    """Return ([(page_number, text), ...], total_pages, skipped_pages)."""
    reader = PdfReader(str(pdf_path))
    total = len(reader.pages)
    limit = min(total, max_pages) if max_pages else total
    pages, skipped = [], 0
    for i in range(limit):
        text = clean_text(reader.pages[i].extract_text() or "")
        if len(text) < MIN_CHARS:
            skipped += 1
            continue
        pages.append((i + 1, text))
    return pages, total, skipped


def chunk_pages(pages):
    """Split each page into overlapping chunks, keeping the page number."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for page_no, text in pages:
        for piece in splitter.split_text(text):
            piece = piece.strip()
            if len(piece) >= MIN_CHARS:
                chunks.append((page_no, piece))
    return chunks


def ingest_pdf(pdf_path, max_pages: int | None = None) -> dict:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise SystemExit(f"File not found: {pdf_path}")

    start = time.time()
    pages, total, skipped = extract_pages(pdf_path, max_pages)
    read = min(total, max_pages) if max_pages else total
    print(f"PDF: {pdf_path.name} | pages read: {read} of {total} | skipped (no text): {skipped}")
    if not pages:
        raise SystemExit("No text found. This looks like a scanned PDF. Use a text-based PDF.")

    chunks = chunk_pages(pages)
    avg = sum(len(c[1]) for c in chunks) // len(chunks)
    print(f"Chunks created: {len(chunks)} (average {avg} characters)")

    print("Creating embeddings...")
    embeddings = embed_texts([c[1] for c in chunks], show_progress=True)

    conn = get_conn()
    with conn.transaction():
        # re-running on the same file replaces the old copy
        conn.execute("DELETE FROM documents WHERE filename = %s", (pdf_path.name,))
        doc_id = conn.execute(
            "INSERT INTO documents (filename, page_count) VALUES (%s, %s) RETURNING id",
            (pdf_path.name, total),
        ).fetchone()[0]
        rows = [
            (doc_id, idx, page_no, text, emb)
            for idx, ((page_no, text), emb) in enumerate(zip(chunks, embeddings))
        ]
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks (document_id, chunk_index, page, content, embedding) "
                "VALUES (%s, %s, %s, %s, %s)",
                rows,
            )
    conn.close()

    seconds = round(time.time() - start, 1)
    print(f"Stored document id {doc_id} with {len(chunks)} chunks in {seconds}s")
    return {"document_id": doc_id, "chunks": len(chunks), "seconds": seconds}