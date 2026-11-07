from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app import config

# bge models work best when SEARCH QUERIES start with this sentence.
# Documents (chunks) are embedded as they are, without it.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Load the embedding model once and reuse it."""
    return SentenceTransformer(config.EMBED_MODEL)


def embed_texts(texts: list[str], batch_size: int = 32, show_progress: bool = False):
    """Embed many chunks. Returns a numpy array of shape (n, 384)."""
    return get_model().encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
    )


def embed_query(query: str):
    """Embed one search question. Returns a numpy array of shape (384,)."""
    return get_model().encode(QUERY_PREFIX + query, normalize_embeddings=True)