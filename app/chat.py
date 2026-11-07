from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI

from app import config


@lru_cache(maxsize=4)
def get_chat_model(temperature: float = 0.0) -> ChatGoogleGenerativeAI:
    """LangChain chat model. Swapping providers later means changing only this function."""
    if not config.GEMINI_API_KEY or not config.GEMINI_MODEL:
        raise SystemExit("Set GEMINI_API_KEY and GEMINI_MODEL in .env")
    return ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL.removeprefix("models/"),
        google_api_key=config.GEMINI_API_KEY,
        temperature=temperature,
        max_retries=4,
    )


def message_text(msg) -> str:
    """Get plain text out of a LangChain message (content can be a string or a list of parts)."""
    content = msg.content
    if isinstance(content, str):
        return content
    parts = []
    for p in content:
        if isinstance(p, str):
            parts.append(p)
        elif isinstance(p, dict) and p.get("type") == "text":
            parts.append(p.get("text", ""))
    return "".join(parts)