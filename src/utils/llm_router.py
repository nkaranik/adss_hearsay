from __future__ import annotations

from typing import Optional

import src.utils.gemini_client as gemini_client
import src.utils.local_llm_client as local_client


def get_provider(cfg: dict | None = None) -> str:
    if not cfg:
        return "gemini"
    return cfg.get("llm", {}).get("provider", "gemini").lower()


def call_text_generation(
    prompt: str,
    cfg: dict | None = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> str:
    provider = get_provider(cfg)

    if provider == "local":
        return local_client.call_local_llm(
            prompt=prompt,
            cfg=cfg,
            max_tokens=max_tokens or 8192,
            temperature=temperature if temperature is not None else 0.0,
        )

    gem = (cfg or {}).get("gemini", {})
    return gemini_client.call_gemini(
        prompt=prompt,
        model=gem.get("model", "gemini-2.5-flash"),
        max_tokens=max_tokens or gem.get("max_tokens", 8192),
        temperature=temperature if temperature is not None else gem.get("temperature", 0.0),
    )