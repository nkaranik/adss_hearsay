"""
Gemini API client with:
  - API key loaded from env OR set at runtime via dashboard
  - Retry with exponential back-off honouring Gemini's retryDelay
  - Thinking-model handling: extracts answer from non-thought parts when
    response.text is empty (Gemini 2.5-flash / 2.5-pro are thinking models)
  - Key diagnostics (masked display)
"""
from __future__ import annotations

import logging
import os
import re
import time
from typing import Optional

logger = logging.getLogger(__name__)

_runtime_api_key: Optional[str] = None
_client_cache: dict = {}


# ── Key management ────────────────────────────────────────────────────────────

def set_runtime_key(key: str) -> None:
    global _runtime_api_key
    _runtime_api_key = key.strip()
    _client_cache.clear()
    logger.info("Runtime API key updated.")


def get_active_key() -> Optional[str]:
    return _runtime_api_key or os.environ.get("GEMINI_API_KEY")


def masked_key(key: Optional[str]) -> str:
    if not key:
        return "❌ NOT SET"
    if len(key) <= 8:
        return "****"
    return f"{key[:4]}…{key[-4:]}"


def get_client():
    key = get_active_key()
    if not key:
        raise EnvironmentError(
            "No Gemini API key found.\n"
            "Either set GEMINI_API_KEY env var or paste it in the dashboard sidebar."
        )
    if _client_cache.get("key") != key:
        import google.genai as genai  # type: ignore
        _client_cache["client"] = genai.Client(api_key=key)
        _client_cache["key"]    = key
        logger.info(f"Gemini client initialised with key {masked_key(key)}")
    return _client_cache["client"]


# ── Response text extraction ──────────────────────────────────────────────────

def _extract_text(response) -> str:
    """
    Extract the answer text from a Gemini response.

    Gemini 2.5-flash and 2.5-pro are thinking models. Their response may have:
      - response.text          → empty string when all output was thinking
      - response.candidates[0].content.parts → list of Part objects,
        some with thought=True (internal reasoning), some with thought=False (answer)

    We prefer the non-thought parts. Fall back to response.text, then full text.
    """
    # 1. Try response.text (works for non-thinking models and simple responses)
    try:
        text = response.text
        if text and text.strip():
            return text.strip()
    except Exception:
        pass

    # 2. Extract non-thought parts from candidates (thinking models)
    try:
        parts = response.candidates[0].content.parts
        answer_parts = []
        thought_parts = []
        for part in parts:
            is_thought = getattr(part, "thought", False)
            part_text  = getattr(part, "text", "") or ""
            if is_thought:
                thought_parts.append(part_text)
            else:
                if part_text.strip():
                    answer_parts.append(part_text)

        if answer_parts:
            return "\n".join(answer_parts).strip()

        # If only thought parts exist, the model reasoned but produced no answer
        if thought_parts:
            logger.warning(
                "Gemini returned only thinking content (no answer part). "
                "The model may have used its full token budget on reasoning. "
                "Consider using gemini-2.0-flash for simpler tasks, or "
                "increase max_tokens."
            )

    except Exception as e:
        logger.debug(f"Part extraction failed: {e}")

    # 3. Last resort: str of response
    try:
        full = str(response)
        if full and full.strip():
            return full.strip()
    except Exception:
        pass

    return ""


# ── Retry helper ──────────────────────────────────────────────────────────────

def _parse_retry_delay(exc: Exception) -> float:
    m = re.search(r"retryDelay['\": ]+(\d+)", str(exc))
    return float(m.group(1)) if m else 0.0


def call_gemini(
    prompt: str,
    model: str = "gemini-2.5-flash",
    max_tokens: int = 4096,
    temperature: float = 0.0,
    max_retries: int = 4,
    base_delay: float = 5.0,
) -> str:
    from google.genai import types  # type: ignore

    last_exc: Exception = RuntimeError("No attempts made.")
    for attempt in range(1, max_retries + 1):
        try:
            client   = get_client()
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )

            text = _extract_text(response)

            if not text:
                logger.warning(
                    f"Gemini returned empty response (model={model}, "
                    f"attempt={attempt}/{max_retries}). "
                    "This can happen when the thinking budget exceeds max_tokens. "
                    "Try gemini-2.0-flash which is not a thinking model."
                )

            return text

        except Exception as exc:
            last_exc  = exc
            err_str   = str(exc)
            is_429    = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str
            is_503    = "503" in err_str or "UNAVAILABLE" in err_str

            if not (is_429 or is_503) or attempt == max_retries:
                raise

            retry_delay = _parse_retry_delay(exc)
            backoff     = base_delay * (2 ** (attempt - 1))
            wait        = max(retry_delay, backoff)

            logger.warning(
                f"Gemini {exc.__class__.__name__} attempt {attempt}/{max_retries}. "
                f"Waiting {wait:.0f}s… (key={masked_key(get_active_key())})"
            )
            time.sleep(wait)

    raise last_exc


def gemini_cfg(cfg: dict) -> dict:
    return cfg.get("gemini", {
        "model": "gemini-2.5-flash",
        "max_tokens": 4096,
        "temperature": 0.0,
    })
