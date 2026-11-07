import os

from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "")
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://truthlens:truthlens@localhost:5432/truthlens"
)
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_DIM = 384  # output size of bge-small