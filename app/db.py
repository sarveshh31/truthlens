import psycopg
from pgvector.psycopg import register_vector

from app import config


def get_conn() -> psycopg.Connection:
    """Open a connection that understands the pgvector 'vector' type."""
    conn = psycopg.connect(config.DATABASE_URL, autocommit=True)
    register_vector(conn)
    return conn


SCHEMA_STATEMENTS = [
    "CREATE EXTENSION IF NOT EXISTS vector",
    """
    CREATE TABLE IF NOT EXISTS documents (
        id          SERIAL PRIMARY KEY,
        filename    TEXT NOT NULL,
        page_count  INTEGER,
        created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    f"""
    CREATE TABLE IF NOT EXISTS chunks (
        id           SERIAL PRIMARY KEY,
        document_id  INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        chunk_index  INTEGER NOT NULL,
        page         INTEGER,
        content      TEXT NOT NULL,
        embedding    vector({config.EMBED_DIM}) NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops)
    """,
    "CREATE INDEX IF NOT EXISTS chunks_document_idx ON chunks (document_id)",
]


def init_db() -> None:
    """Create the extension, tables and indexes. Safe to run many times."""
    # plain connection first: the 'vector' type does not exist until the extension is created
    with psycopg.connect(config.DATABASE_URL, autocommit=True) as conn:
        for stmt in SCHEMA_STATEMENTS:
            conn.execute(stmt)