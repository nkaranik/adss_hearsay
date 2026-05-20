"""
Unified LLM router.
cfg["backend"] = "gemini"   → Google Gemini API
cfg["backend"] = "lmstudio" → Local LM Studio (OpenAI-compatible)
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def get_backend(cfg: dict) -> str:
    return cfg.get("backend", "gemini").lower()


def llm_model(cfg: dict) -> str:
    if get_backend(cfg) == "lmstudio":
        return cfg.get("lmstudio", {}).get("model", "qwen/qwen3-30b-a3b")
    return cfg.get("gemini", {}).get("model", "gemini-2.5-flash")


def backend_label(cfg: dict) -> str:
    b = get_backend(cfg)
    if b == "lmstudio":
        url      = cfg.get("lmstudio", {}).get("base_url", "http://127.0.0.1:1234/v1")
        thinking = "thinking OFF" if cfg.get("lmstudio", {}).get("disable_thinking", True) else "thinking ON"
        return f"LM Studio ({llm_model(cfg)}, {thinking} @ {url})"
    return f"Gemini ({llm_model(cfg)})"


def call_llm(prompt: str, cfg: dict, max_tokens: int | None = None) -> str:
    """Route a prompt to the configured backend."""
    b = get_backend(cfg)

    if b == "lmstudio":
        import src.utils.lmstudio_client as _ls
        lcfg = _ls.lmstudio_cfg(cfg)
        return _ls.call_lmstudio(
            prompt,
            model=lcfg.get("model", "qwen/qwen3-30b-a3b"),
            max_tokens=max_tokens or lcfg.get("max_tokens", 4096),
            temperature=lcfg.get("temperature", 0.0),
            base_url=lcfg.get("base_url", "http://127.0.0.1:1234/v1"),
            disable_thinking=lcfg.get("disable_thinking", True),
        )

    # Default: Gemini
    import src.utils.gemini_client as _gc
    gcfg = _gc.gemini_cfg(cfg)
    return _gc.call_gemini(
        prompt,
        model=gcfg.get("model", "gemini-2.5-flash"),
        max_tokens=max_tokens or gcfg.get("max_tokens", 4096),
        temperature=gcfg.get("temperature", 0.0),
    )
