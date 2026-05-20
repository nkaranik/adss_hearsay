from __future__ import annotations

import logging
import os
import time
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

_runtime_base_url: Optional[str] = None
_runtime_model: Optional[str] = None

def set_runtime_local(base_url: str, model: str) -> None:
    global _runtime_base_url, _runtime_model
    _runtime_base_url = base_url.rstrip("/")
    _runtime_model = model.strip()
    logger.info("Local LLM runtime config updated.")

def get_local_base_url(cfg: dict | None = None) -> str:
    if _runtime_base_url:
        return _runtime_base_url
    if cfg:
        return cfg.get("llm", {}).get("local", {}).get("base_url", "http://127.0.0.1:1234/v1")
    return os.environ.get("LOCAL_LLM_BASE_URL", "http://127.0.0.1:1234/v1")

def get_local_model(cfg: dict | None = None) -> str:
    if _runtime_model:
        return _runtime_model
    if cfg:
        return cfg.get("llm", {}).get("local", {}).get("model", "qwen/qwen3.6-35b-a3b")
    return os.environ.get("LOCAL_LLM_MODEL", "qwen/qwen3.6-35b-a3b")

def call_local_llm(
    prompt: str,
    cfg: dict | None = None,
    max_tokens: int = 8192,
    temperature: float = 0.0,
    max_retries: int = 3,
    base_delay: float = 2.0,
    timeout: float = 180.0,
) -> str:
    """
    OpenAI-compatible call to LM Studio local server.
    """
    base_url = get_local_base_url(cfg)
    model = get_local_model(cfg)

    url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return only the requested output. If JSON is requested, return valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

    last_exc: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            with httpx.Client(timeout=timeout) as client:
                r = client.post(url, json=payload)
                r.raise_for_status()
                data = r.json()
                return data["choices"][0]["message"]["content"]
        except Exception as exc:
            last_exc = exc
            if attempt >= max_retries:
                break
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning(
                "Local LLM error attempt %s/%s. Waiting %.1fs… url=%s model=%s err=%s",
                attempt, max_retries, delay, base_url, model, str(exc)[:200],
            )
            time.sleep(delay)

    raise RuntimeError(f"Local LLM call failed after {max_retries} attempts: {last_exc}")