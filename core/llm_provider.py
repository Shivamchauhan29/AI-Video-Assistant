import os

import groq
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_core.runnables import RunnableLambda

from core.pipeline_logger import log_if

GROQ_MODEL = "openai/gpt-oss-120b"
MISTRAL_MODEL = "mistral-small-latest"

# Groq failures that should fall back to Mistral: rate-limit/quota exhaustion
# (429 — Groq's SDK raises the same RateLimitError for both the per-minute
# and per-day quota variants, distinguished only by message text), request
# timeouts, connection errors, and 5xx server errors. Anything else (e.g. a
# genuine bad request) is a real bug and should surface normally rather than
# being silently masked by a fallback.
GROQ_FALLBACK_EXCEPTIONS = (
    groq.RateLimitError,
    groq.APITimeoutError,
    groq.APIConnectionError,
    groq.InternalServerError,
)


def _describe_groq_failure(exc: Exception) -> str:
    """Distinguish why Groq failed, for the fallback log line."""
    if isinstance(exc, groq.RateLimitError):
        body = getattr(exc, "body", None) or {}
        message = ""
        if isinstance(body, dict):
            message = str(body.get("error", {}).get("message", ""))
        message = message or str(exc)
        if "per day" in message.lower() or "tpd" in message.lower():
            return "quota exhausted (daily token limit)"
        return "rate limit hit"
    if isinstance(exc, groq.APITimeoutError):
        return "request timed out"
    if isinstance(exc, groq.APIConnectionError):
        return "connection error"
    if isinstance(exc, groq.InternalServerError):
        return "server error (5xx)"
    return f"{type(exc).__name__}: {exc}"


def get_llm(temperature: float = 0.2, logger=None):
    """
    Groq-primary / Groq-secondary / Mistral-fallback LLM, shared by every
    chain (summarizer, extractor, highlighter, rag_engine) so this fallback
    logic lives in one place instead of being duplicated per module.

    Tries GROQ_API_KEY first; if that hits a rate-limit/quota error, timeout,
    connection error, or 5xx, tries GROQ_API_KEY_2 (a second Groq key/account,
    if configured — omitting it just skips straight to Mistral, so this
    stays backward compatible with single-key setups); if that also fails
    the same way, falls back to Mistral. Each hop logs which reason
    triggered it. A Mistral failure (the final fallback failing) is not
    special-cased — it propagates normally, same as before.
    """
    providers = []

    groq_key_1 = os.getenv("GROQ_API_KEY")
    if groq_key_1:
        providers.append(("Groq primary key", ChatGroq(model=GROQ_MODEL, groq_api_key=groq_key_1, temperature=temperature)))

    groq_key_2 = os.getenv("GROQ_API_KEY_2")
    if groq_key_2:
        providers.append(("Groq secondary key", ChatGroq(model=GROQ_MODEL, groq_api_key=groq_key_2, temperature=temperature)))

    mistral_llm = ChatMistralAI(model=MISTRAL_MODEL, mistral_api_key=os.getenv("MISTRAL_API_KEY"), temperature=temperature)

    def _invoke_with_fallback(input):
        for i, (name, model) in enumerate(providers):
            try:
                return model.invoke(input)
            except GROQ_FALLBACK_EXCEPTIONS as e:
                reason = _describe_groq_failure(e)
                next_name = providers[i + 1][0] if i + 1 < len(providers) else "Mistral"
                log_if(logger, "Groq", f"{reason} on {name}, falling back to {next_name}")
        return mistral_llm.invoke(input)

    return RunnableLambda(_invoke_with_fallback)
