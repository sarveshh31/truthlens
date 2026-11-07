import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def step(msg):
    print(f"\n=== {msg} ===")


step("1. Python version")
print(sys.version)
if sys.version_info < (3, 10):
    sys.exit("FAIL: need Python 3.10+ (we use 3.11)")
print("OK")

step("2. Config")
from app import config

if not config.GEMINI_API_KEY or "paste" in config.GEMINI_API_KEY.lower():
    sys.exit("FAIL: put your real Gemini key in .env")
print("OK (key loaded, hidden)")

step("3. Libraries")
import psycopg  # noqa: F401
import pgvector  # noqa: F401
import pypdf  # noqa: F401
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: F401

print("OK")

step("4. Embedding model (first run downloads about 130 MB)")
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(config.EMBED_MODEL)
vec = model.encode("TruthLens checks every sentence.", normalize_embeddings=True)
print("embedding size:", len(vec))
if len(vec) != config.EMBED_DIM:
    sys.exit(f"FAIL: expected {config.EMBED_DIM}")
print("OK")

step("5. Gemini API")
from google import genai

client = genai.Client(api_key=config.GEMINI_API_KEY)
names = []
for m in client.models.list():
    actions = getattr(m, "supported_actions", None) or []
    if "generateContent" in actions:
        names.append(m.name)
if not names:
    names = [m.name for m in client.models.list()]
print("Flash-type models available to you:")
for n in names:
    if "flash" in n.lower():
        print("  ", n)

if not config.GEMINI_MODEL:
    print("\nNow set GEMINI_MODEL in .env to one name from the list above, then run this script again.")
    sys.exit(0)

resp = client.models.generate_content(
    model=config.GEMINI_MODEL, contents="Reply with exactly: TruthLens online"
)
print("Reply:", resp.text)
print("OK")
print("\nALL CHECKS PASSED")