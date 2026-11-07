
import time
import random
from functools import lru_cache

from google import genai
from google.genai import errors, types

from app import config


@lru_cache(maxsize=1)
def _client() -> genai.Client:
    if not config.GEMINI_API_KEY:
        raise SystemExit("GEMINI_API_KEY is missing in .env")
    return genai.Client(api_key=config.GEMINI_API_KEY)


def generate(prompt: str, temperature: float = 0.2) -> str:
    """Call Gemini with retries for temporary API errors."""

    if not config.GEMINI_MODEL:
        raise SystemExit("GEMINI_MODEL is empty in .env")

    for attempt in range(4):
        try:
            resp = _client().models.generate_content(
                model=config.GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature
                ),
            )
            return resp.text or ""

        except errors.APIError as e:
            if e.code in (429, 500, 503) and attempt < 3:
                wait = 5 * 2**attempt + random.uniform(0, 2)

                print(
                    f"  (Gemini temporarily unavailable: "
                    f"{e.code}. Retrying in {wait:.1f}s...)"
                )

                time.sleep(wait)
                continue

            if e.code in (429, 500, 503):
                raise RuntimeError(
                    f"Gemini is currently unavailable (HTTP {e.code}). "
                    "Please try again later."
                ) from e

            raise

    return ""
