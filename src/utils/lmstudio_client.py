"""
LM Studio client — OpenAI-compatible local API.
Default endpoint: http://127.0.0.1:1234/v1  (no API key needed).

Token budget problem with Qwen3
────────────────────────────────
Qwen3.6-35B reasons internally even when thinking is "disabled".
With 8192 context and ~674 prompt tokens, 7518 tokens are available
for completion. If max_tokens=4096, the model spends ~3400 on reasoning
and only ~678 on the actual JSON — not enough for 6 arguments.

Fix: set max_tokens close to the full available context (e.g. 7000).
This gives the model room for both reasoning AND the full JSON answer.

Rule of thumb:
  max_tokens = context_length - prompt_tokens - 500 (safety buffer)
  With 8192 context: max_tokens ≈ 7000
  With 4096 context: max_tokens ≈ 3200  (too small — upgrade context first)
"""
from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)

_client_cache: dict = {}


class ContextTooSmallError(Exception):
    """Raised when context window is too small to produce a complete answer."""
    pass


def get_client(base_url: str = "http://127.0.0.1:1234/v1"):
    if _client_cache.get("url") != base_url:
        from openai import OpenAI  # type: ignore
        _client_cache["client"] = OpenAI(base_url=base_url, api_key="lm-studio")
        _client_cache["url"]    = base_url
        logger.info(f"LM Studio client initialised at {base_url}")
    return _client_cache["client"]


def call_lmstudio(
    prompt:           str,
    model:            str   = "qwen/qwen3.6-35b-a3b",
    max_tokens:       int   = 7000,
    temperature:      float = 0.0,
    base_url:         str   = "http://127.0.0.1:1234/v1",
    max_retries:      int   = 3,
    base_delay:       float = 3.0,
    disable_thinking: bool  = True,
) -> str:
    """
    Call a local LM Studio model.

    Key: max_tokens should be ≈ context_length - prompt_tokens - 500.
    For 8192 context → use 7000. For 4096 context → upgrade context first.
    """
    # Prompt-level thinking signals
    if disable_thinking:
        if "/no_think" not in prompt:
            prompt = prompt.rstrip() + "\n\n/no_think"
    else:
        if "/think" not in prompt and "/no_think" not in prompt:
            prompt = prompt.rstrip() + "\n\n/think"

    client    = get_client(base_url)
    last_exc: Exception = RuntimeError("No attempts made.")

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                extra_body={
                    "chat_template_kwargs": {
                        "enable_thinking": not disable_thinking
                    }
                },
            )

            choice = resp.choices[0]
            text   = (choice.message.content or "").strip()

            # ── Token diagnostics ──────────────────────────────────────────
            usage             = getattr(resp, "usage", None)
            prompt_tok        = getattr(usage, "prompt_tokens",     0) if usage else 0
            completion_tok    = getattr(usage, "completion_tokens", 0) if usage else 0
            total_tok         = getattr(usage, "total_tokens",      0) if usage else 0
            details           = getattr(usage, "completion_tokens_details", None)
            reasoning_tok     = getattr(details, "reasoning_tokens", 0) if details else 0
            answer_tok        = completion_tok - reasoning_tok
            finish_reason     = choice.finish_reason

            logger.info(
                f"Tokens — prompt: {prompt_tok} | "
                f"reasoning: {reasoning_tok} | answer: {answer_tok} | "
                f"total: {total_tok} | finish: {finish_reason}"
            )

            # ── Truncation detection ───────────────────────────────────────
            # finish_reason="length" with answer_tok < 200 means the JSON
            # was cut off — model ran out of budget.
            if finish_reason == "length" and answer_tok < 200 and not text:
                raise ContextTooSmallError(
                    f"Context overflow: reasoning used {reasoning_tok} tokens, "
                    f"leaving only {answer_tok} for the JSON answer.\n\n"
                    f"► Fix: In LM Studio → loaded model → ⚙️ gear → "
                    f"set Context Length to 8192 → Apply.\n"
                    f"► Also check that max_tokens in the sidebar is set to ~7000."
                )

            # Warn if truncated but we have partial content
            if finish_reason == "length" and text:
                logger.warning(
                    f"Response truncated (finish=length). "
                    f"answer_tokens={answer_tok}, reasoning_tokens={reasoning_tok}. "
                    f"JSON repair will attempt to recover partial output. "
                    f"For complete output, increase context to 16384 in LM Studio."
                )

            # ── Fallback to reasoning_content if content empty ─────────────
            if not text:
                rc = getattr(choice.message, "reasoning_content", None)
                if rc and rc.strip():
                    logger.warning(
                        f"content empty (answer_tok={answer_tok}); "
                        "trying reasoning_content as fallback."
                    )
                    text = rc.strip()

            return text

        except ContextTooSmallError:
            raise

        except Exception as exc:
            last_exc = exc
            err      = str(exc)
            is_transient = any(c in err for c in ["503", "429", "Connection", "timeout"])
            if not is_transient or attempt == max_retries:
                raise
            wait = base_delay * (2 ** (attempt - 1))
            logger.warning(
                f"LM Studio error attempt {attempt}/{max_retries}. "
                f"Waiting {wait:.0f}s…"
            )
            time.sleep(wait)

    raise last_exc


def lmstudio_cfg(cfg: dict) -> dict:
    return cfg.get("lmstudio", {
        "model":            "qwen/qwen3.6-35b-a3b",
        "base_url":         "http://127.0.0.1:1234/v1",
        "max_tokens":       7000,
        "temperature":      0.0,
        "disable_thinking": True,
    })
