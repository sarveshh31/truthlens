from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI

from app import config


@lru_cache(maxsize=4)
def get_chat_model(temperature: float = 0.0):
    """The only place that knows which LLM provider is used."""
    if config.LLM_PROVIDER == "groq":
        if not config.GROQ_API_KEY or not config.GROQ_MODEL:
            raise SystemExit("Set GROQ_API_KEY and GROQ_MODEL in .env")
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=config.GROQ_MODEL,
            api_key=config.GROQ_API_KEY,
            temperature=temperature,
            max_retries=2,
        )

    if not config.GEMINI_API_KEY or not config.GEMINI_MODEL:
        raise SystemExit("Set GEMINI_API_KEY and GEMINI_MODEL in .env")
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL.removeprefix("models/"),
        google_api_key=config.GEMINI_API_KEY,
        temperature=temperature,
        max_retries=2,
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