"""
Module A+B combined: single LLM call extracts arguments AND scores strength.
Now fully domain-generic — works for any decision problem, not just hearsay.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from src.data.models import (
    Argument, ArgumentStrength, ClaimNode,
    ExtractionResult, Relation, RelationType, RubricScore,
)
import src.utils.llm_client as _llm

logger = logging.getLogger(__name__)

_COMBINED_PATH = Path("prompts/combined.txt")
_EXTRACT_PATH  = Path("prompts/extraction.txt")

_WEIGHTS = {
    "legal_relevance":        0.30,
    "factual_grounding":      0.20,
    "specificity":            0.15,
    "logical_coherence":      0.15,
    "fre_801c_applicability": 0.20,
}

_DEFAULT_DECISION = "Is the statement described in this narrative hearsay under FRE 801(c)?"


def _load_template() -> str:
    for p in (_COMBINED_PATH, _EXTRACT_PATH):
        if p.exists():
            return p.read_text()
    return (
        "Extract arguments as a JSON object for this decision problem:\n"
        "{decision_problem}\n\nCase: {case_id}\nMax args: {max_arguments}\n\n"
        "Narrative:\n{narrative}"
    )


def _render(template: str, case_id: str, narrative: str,
            max_arguments: int, decision_problem: str) -> str:
    r = template.replace("{case_id}", case_id)
    r = r.replace("{narrative}", narrative)
    r = r.replace("{max_arguments}", str(max_arguments))
    r = r.replace("{decision_problem}", decision_problem)
    return r


def _strip_fences(text: str) -> str:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()


def _repair_json(raw: str) -> dict[str, Any] | None:
    cleaned = _strip_fences(raw)
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    try:
        from json_repair import repair_json  # type: ignore
        r = repair_json(cleaned, return_objects=True)
        if isinstance(r, dict):
            return r
        if isinstance(r, list) and r and isinstance(r[0], dict):
            return r[0]
    except Exception as e:
        logger.debug(f"json_repair: {e}")
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
        try:
            from json_repair import repair_json  # type: ignore
            r = repair_json(m.group(0), return_objects=True)
            if isinstance(r, dict):
                return r
        except Exception:
            pass
    return None


def _normalise_stance(v: str) -> str | None:
    """
    'support' → supports the claim being true
    'attack'  → argues the claim is false / exception applies
    None      → neutral / irrelevant → excluded from QBAF graph
    """
    v = v.lower().strip()
    if "support" in v:
        return "support"
    if "attack" in v or "oppose" in v or "against" in v:
        return "attack"
    return None   # neutral, n/a, irrelevant, etc.


def _normalise_rel_type(v: str) -> str:
    v = v.lower()
    if "support" in v:
        return "support"
    if "attack" in v:
        return "attack"
    return "neutral"


def _parse_combined(
    raw: str,
    case_id: str,
    input_text: str,
    decision_problem: str,
    min_confidence: float,
    include_neutral: bool,
) -> tuple[ExtractionResult, list[ArgumentStrength]]:
    data = _repair_json(raw)
    if data is None:
        logger.warning(f"[{case_id}] JSON repair failed. Snippet: {raw[:300]!r}")
        return (
            ExtractionResult(
                case_id=case_id, input_text=input_text,
                claim=ClaimNode(text=decision_problem),
                raw_llm_response=raw,
                parse_error=f"JSON repair failed. Raw: {raw[:400]}",
            ),
            [],
        )

    args:            list[Argument]         = []
    strengths:       list[ArgumentStrength] = []
    seen:            set[str]               = set()
    n_raw_arguments: int                    = 0   # count before neutral filter

    for raw_arg in data.get("arguments", []):
        try:
            rubric_data   = raw_arg.pop("rubric",        None)
            tau_val       = raw_arg.pop("tau",           None)
            justification = raw_arg.pop("justification", "")

            n_raw_arguments += 1

            raw_stance = raw_arg.get("stance_to_claim", "")
            normalised = _normalise_stance(str(raw_stance))

            if normalised is None:
                logger.debug(
                    f"[{case_id}] '{raw_arg.get('id','?')}' stance='{raw_stance}' "
                    "is neutral — excluded from graph."
                )
                continue

            raw_arg["stance_to_claim"] = normalised
            if raw_arg.get("confidence", 1.0) < min_confidence:
                continue

            arg = Argument(**raw_arg)
            while arg.id in seen:
                arg = arg.model_copy(update={"id": f"{arg.id}_x"})
            seen.add(arg.id)
            args.append(arg)

            if rubric_data and isinstance(rubric_data, dict):
                try:
                    # Normalise any camelCase or missing rubric keys from LLM output
                    key_map = {
                        "legalrelevance": "legal_relevance",
                        "factualgrounding": "factual_grounding",
                        "logicalcoherence": "logical_coherence",
                        "fre801capplicability": "fre_801c_applicability",
                        "freapplicability": "fre_801c_applicability",
                    }
                    norm = {}
                    for k, v in rubric_data.items():
                        clean = k.lower().replace(" ", "").replace("-", "").replace("_", "")
                        norm[key_map.get(clean, k)] = v
                    for field in _WEIGHTS:
                        if field not in norm:
                            norm[field] = float(arg.confidence)
                    rubric = RubricScore(**norm)
                    if tau_val is None:
                        tau_val = sum(
                            getattr(rubric, d) * w for d, w in _WEIGHTS.items()
                        )
                    strengths.append(ArgumentStrength(
                        argument_id=arg.id,
                        tau=max(0.1, min(1.0, float(tau_val))),
                        rubric=rubric,
                        justification=str(justification),
                        model="combined",
                    ))
                except Exception as e:
                    logger.info(f"Rubric parse error for {arg.id}: {e} — using fallback")
                    _fallback_strength(strengths, arg)
            else:
                logger.debug(f"No rubric for {arg.id} — using fallback")
                _fallback_strength(strengths, arg)

        except Exception as e:
            logger.debug(f"[{case_id}] Skipping malformed argument: {e}")

    n_neutral = n_raw_arguments - len(args)
    if n_neutral > 0:
        logger.info(
            f"[{case_id}] {n_neutral}/{n_raw_arguments} argument(s) were neutral "
            "and excluded from the graph."
        )

    valid = {a.id for a in args} | {"phi"}
    rels: list[Relation] = []
    for raw_rel in data.get("relations", []):
        try:
            if isinstance(raw_rel.get("type"), str):
                raw_rel["type"] = _normalise_rel_type(raw_rel["type"])
            rel = Relation(**raw_rel)
            if rel.source not in valid or rel.target not in valid:
                continue
            if rel.type == RelationType.NEUTRAL and not include_neutral:
                continue
            rels.append(rel)
        except Exception as e:
            logger.debug(f"[{case_id}] Skipping malformed relation: {e}")

    try:
        extraction = ExtractionResult(
            case_id=case_id,
            input_text=input_text,
            claim=ClaimNode(text=decision_problem),
            arguments=args,
            relations=rels,
            raw_llm_response=raw,
            n_raw_arguments=n_raw_arguments,
        )
    except Exception as e:
        logger.warning(f"[{case_id}] Extraction validation error: {e}")
        extraction = ExtractionResult(
            case_id=case_id,
            input_text=input_text,
            claim=ClaimNode(text=decision_problem),
            raw_llm_response=raw,
            parse_error=str(e),
            arguments=args,
            n_raw_arguments=n_raw_arguments,
        )

    return extraction, strengths


def _fallback_strength(strengths: list[ArgumentStrength], arg: Argument) -> None:
    tau = max(0.1, min(1.0, arg.confidence))
    strengths.append(ArgumentStrength(
        argument_id=arg.id, tau=tau,
        rubric=RubricScore(**{d: tau for d in _WEIGHTS}),
        justification="Fallback: rubric not provided by LLM.",
        model="fallback",
    ))


class ArgumentExtractor:
    def __init__(self, cfg: dict):
        self.cfg         = cfg
        mcfg             = cfg.get("mining", {})
        self.max_args    = mcfg.get("max_arguments", 8)
        self.min_conf    = mcfg.get("min_confidence", 0.3)
        self.inc_neutral = mcfg.get("include_neutral_relations", False)
        self.max_retries = mcfg.get("max_retries", 2)
        self._tmpl       = _load_template()

    def _prompt(self, case_id: str, narrative: str, decision_problem: str) -> str:
        return _render(self._tmpl, case_id, narrative,
                       self.max_args, decision_problem)

    def extract_combined(
        self,
        case_id:          str,
        narrative:        str,
        decision_problem: str = _DEFAULT_DECISION,
    ) -> tuple[ExtractionResult, list[ArgumentStrength]]:
        prompt = self._prompt(case_id, narrative, decision_problem)
        raw    = ""
        for attempt in range(1, self.max_retries + 1):
            try:
                raw = _llm.call_llm(prompt, self.cfg)
                logger.debug(f"[{case_id}] Raw (first 400): {raw[:400]}")
                extraction, strengths = _parse_combined(
                    raw, case_id, narrative, decision_problem,
                    self.min_conf, self.inc_neutral,
                )
                if extraction.parse_error and attempt < self.max_retries:
                    logger.warning(f"[{case_id}] Parse failed, retrying…")
                    continue
                return extraction, strengths
            except Exception as exc:
                logger.error(f"[{case_id}] LLM error attempt {attempt}: {exc}")
                if attempt == self.max_retries:
                    return (
                        ExtractionResult(
                            case_id=case_id, input_text=narrative,
                            claim=ClaimNode(text=decision_problem),
                            raw_llm_response=raw, parse_error=str(exc),
                        ),
                        [],
                    )
        return (
            ExtractionResult(
                case_id=case_id, input_text=narrative,
                claim=ClaimNode(text=decision_problem),
                raw_llm_response=raw,
            ),
            [],
        )
