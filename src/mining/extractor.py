"""
Module A+B combined: single LLM call extracts arguments AND scores strength.
Domain-generic version: the decision problem (phi) is passed through the
pipeline, and if the LLM returns a better `phi` field, that value is adopted.
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

_GENERIC_COMBINED_PATH = Path("prompts/generic_combined.txt")
_COMBINED_PATH = Path("prompts/combined.txt")
_EXTRACT_PATH = Path("prompts/extraction.txt")

# Generic conceptual weights. These are mapped to whatever RubricScore fields
# your current src.data.models.RubricScore actually defines.
_WEIGHTS = {
    "relevance": 0.30,
    "factual_grounding": 0.20,
    "specificity": 0.15,
    "logical_coherence": 0.15,
    "applicability": 0.20,
}

_DEFAULT_DECISION = ""
_GENERIC_FALLBACK_DECISION = "Infer the main yes/no decision from the case."


def _load_template() -> str:
    """Prefer the domain-generic prompt, but keep old prompts as fallback."""
    for p in (_GENERIC_COMBINED_PATH, _COMBINED_PATH, _EXTRACT_PATH):
        if p.exists():
            return p.read_text(encoding="utf-8")
    return (
        "You are a generic decision-support assistant.\n"
        "Decision question: {decision_problem}\n"
        "Case ID: {case_id}\n"
        "Case text:\n{narrative}\n"
        "Extract up to {max_arguments} arguments as JSON.\n"
        "Return a JSON object with fields: case_id, phi, arguments, relations."
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
        repaired = repair_json(cleaned, return_objects=True)
        if isinstance(repaired, dict):
            return repaired
        if isinstance(repaired, list) and repaired and isinstance(repaired[0], dict):
            return repaired[0]
    except Exception as e:
        logger.debug(f"json_repair failed on full text: {e}")

    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        snippet = m.group(0)
        try:
            return json.loads(snippet)
        except Exception:
            pass
        try:
            from json_repair import repair_json  # type: ignore
            repaired = repair_json(snippet, return_objects=True)
            if isinstance(repaired, dict):
                return repaired
        except Exception as e:
            logger.debug(f"json_repair failed on JSON-looking snippet: {e}")

    return None


def _normalise_stance(v: str) -> str | None:
    """
    Return support/attack/None.
    The LLM often emits variants such as yes/no/pro/con/favor/against.
    """
    v = (v or "").lower().strip()
    if not v:
        return None
    if any(x in v for x in ("support", "yes", "favor", "favour", "pro", "for", "positive")):
        return "support"
    if any(x in v for x in ("attack", "no", "oppose", "against", "con", "negative", "undermine")):
        return "attack"
    if "neutral" in v or "irrelevant" in v or "background" in v:
        return None
    return None


def _normalise_rel_type(v: str) -> str:
    v = (v or "").lower()
    if "support" in v or v in {"yes", "for", "pro"}:
        return "support"
    if "attack" in v or "oppose" in v or "against" in v or v in {"no", "con"}:
        return "attack"
    return "neutral"


def _model_fields(model_cls) -> set[str]:
    """Pydantic v2/v1 compatible field-name helper."""
    if hasattr(model_cls, "model_fields"):
        return set(model_cls.model_fields.keys())
    if hasattr(model_cls, "__fields__"):
        return set(model_cls.__fields__.keys())
    return set()


def _clean_key(k: str) -> str:
    return k.lower().replace(" ", "").replace("-", "").replace("_", "")


def _generic_rubric_values(rubric_data: dict[str, Any] | None, confidence: float) -> dict[str, float]:
    """Normalise LLM rubric output to generic conceptual keys."""
    values = {k: float(confidence) for k in _WEIGHTS}
    if not isinstance(rubric_data, dict):
        return values

    key_map = {
        "relevance": "relevance",
        "legalrelevance": "relevance",
        "domainrelevance": "relevance",
        "factualgrounding": "factual_grounding",
        "grounding": "factual_grounding",
        "specificity": "specificity",
        "logicalcoherence": "logical_coherence",
        "coherence": "logical_coherence",
        "applicability": "applicability",
        "legalapplicability": "applicability",
        "fre801capplicability": "applicability",
        "freapplicability": "applicability",
    }

    for k, v in rubric_data.items():
        target = key_map.get(_clean_key(str(k)))
        if target in values:
            try:
                values[target] = max(0.0, min(1.0, float(v)))
            except Exception:
                pass
    return values


def _make_rubric_score(generic_values: dict[str, float]) -> RubricScore:
    """
    Build RubricScore even if your data model still has the old hearsay-specific
    fields. This keeps the generic prompt compatible with the existing model.
    """
    fields = _model_fields(RubricScore)
    relevance = generic_values.get("relevance", 0.5)
    factual = generic_values.get("factual_grounding", 0.5)
    specificity = generic_values.get("specificity", 0.5)
    coherence = generic_values.get("logical_coherence", 0.5)
    applicability = generic_values.get("applicability", 0.5)

    if fields and {"relevance", "factual_grounding", "specificity", "logical_coherence", "applicability"}.issubset(fields):
        payload = {
            "relevance": relevance,
            "factual_grounding": factual,
            "specificity": specificity,
            "logical_coherence": coherence,
            "applicability": applicability,
        }
    elif fields and {"legal_relevance", "factual_grounding", "specificity", "logical_coherence", "fre_801c_applicability"}.issubset(fields):
        # Backward compatibility with the original hearsay model names.
        payload = {
            "legal_relevance": relevance,
            "factual_grounding": factual,
            "specificity": specificity,
            "logical_coherence": coherence,
            "fre_801c_applicability": applicability,
        }
    else:
        # Last resort: try generic names.
        payload = generic_values

    return RubricScore(**payload)


def _tau_from_generic_rubric(values: dict[str, float]) -> float:
    return sum(values.get(k, 0.5) * w for k, w in _WEIGHTS.items())


def _ensure_argument_schema(raw_arg: dict[str, Any]) -> dict[str, Any]:
    """
    Make generic LLM outputs compatible with the existing Argument model.
    If the LLM returns `principle`, `rule`, or `reasoning_principle`, map it to
    `legal_rule` because the original model likely still expects legal_rule.
    """
    raw_arg = dict(raw_arg)
    if "legal_rule" not in raw_arg:
        for alt in ("principle", "rule", "reasoning_principle", "domain_rule", "applicable_principle"):
            if raw_arg.get(alt):
                raw_arg["legal_rule"] = raw_arg[alt]
                break
    if "legal_rule" not in raw_arg:
        raw_arg["legal_rule"] = "Relevant decision principle"
    if "evidence_span" not in raw_arg:
        raw_arg["evidence_span"] = None
    if "confidence" not in raw_arg:
        raw_arg["confidence"] = 0.8
    return raw_arg


def _infer_stance_from_text(raw_arg: dict[str, Any], decision_problem: str) -> str | None:
    """
    Conservative fallback for cases where the LLM labels a real listed argument as neutral.
    If the text clearly contains liability/compensation/breach/damage language, treat it
    as support. If it clearly denies the claim, treat it as attack.
    """
    joined = " ".join(str(raw_arg.get(k, "")) for k in ("text", "legal_rule", "principle", "justification"))
    t = joined.lower()
    dp = decision_problem.lower()

    attack_markers = [
        "not liable", "no liability", "no compensation", "does not deserve",
        "did not breach", "properly performed", "no damage", "not caused",
        "defense", "defence", "excuse", "justified delay",
    ]
    support_markers = [
        "deserves compensation", "entitled to compensation", "liable", "breach",
        "failed", "not finish", "late", "damage", "damaged", "leaked", "poor service",
        "did not perform", "paid as agreed", "caused", "loss", "harm", "defective",
    ]

    if any(m in t for m in attack_markers):
        return "attack"
    if any(m in t for m in support_markers):
        return "support"

    # If the decision itself is a compensation/liability question and the input
    # explicitly calls this item an argument, default to support rather than dropping it.
    if any(m in dp for m in ("compensation", "liable", "liability", "breach", "damages")):
        return "support"

    return None


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

    # IMPORTANT: If the generic prompt asked the LLM to infer phi, adopt it.
    llm_phi = str(data.get("phi", "")).strip()
    if llm_phi:
        decision_problem = llm_phi
        logger.info(f"[{case_id}] Adopted LLM phi: {decision_problem}")

    args: list[Argument] = []
    strengths: list[ArgumentStrength] = []
    seen: set[str] = set()
    n_raw_arguments = 0
    neutral_ids: set[str] = set()

    for raw_arg_in in data.get("arguments", []):
        try:
            if not isinstance(raw_arg_in, dict):
                continue
            raw_arg = dict(raw_arg_in)
            rubric_data = raw_arg.pop("rubric", None)
            tau_val = raw_arg.pop("tau", None)
            justification = raw_arg.pop("justification", "")
            n_raw_arguments += 1

            raw_stance = raw_arg.get("stance_to_claim", "")
            normalised = _normalise_stance(str(raw_stance))
            if normalised is None:
                normalised = _infer_stance_from_text(raw_arg, decision_problem)

            if normalised is None:
                neutral_ids.add(str(raw_arg.get("id", f"arg_{n_raw_arguments}")))
                logger.debug(f"[{case_id}] '{raw_arg.get('id','?')}' stance='{raw_stance}' is neutral — excluded.")
                continue

            raw_arg["stance_to_claim"] = normalised
            raw_arg = _ensure_argument_schema(raw_arg)

            try:
                conf = float(raw_arg.get("confidence", 1.0))
            except Exception:
                conf = 1.0
                raw_arg["confidence"] = conf
            if conf < min_confidence:
                continue

            arg = Argument(**raw_arg)
            while arg.id in seen:
                arg = arg.model_copy(update={"id": f"{arg.id}_x"})
            seen.add(arg.id)
            args.append(arg)

            generic_rubric = _generic_rubric_values(rubric_data, arg.confidence)
            rubric = _make_rubric_score(generic_rubric)
            if tau_val is None:
                tau_val = _tau_from_generic_rubric(generic_rubric)

            strengths.append(ArgumentStrength(
                argument_id=arg.id,
                tau=max(0.1, min(1.0, float(tau_val))),
                rubric=rubric,
                justification=str(justification),
                model="combined",
            ))
        except Exception as e:
            logger.debug(f"[{case_id}] Skipping malformed argument: {e}")

    n_neutral = len(neutral_ids)
    if n_neutral > 0:
        logger.info(f"[{case_id}] {n_neutral}/{n_raw_arguments} argument(s) were neutral and excluded from the graph.")

    valid = {a.id for a in args} | {"phi"}
    rels: list[Relation] = []
    for raw_rel_in in data.get("relations", []):
        try:
            if not isinstance(raw_rel_in, dict):
                continue
            raw_rel = dict(raw_rel_in)
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

    # Ensure every argument has a direct stance edge to phi. This makes the QBAF
    # useful even when the LLM omits relations or only gives arg-arg relations.
    existing = {(r.source, r.target) for r in rels}
    for arg in args:
        if (arg.id, "phi") not in existing:
            rels.append(Relation(
                source=arg.id,
                target="phi",
                type=RelationType.SUPPORT if arg.stance_to_claim.value == "support" else RelationType.ATTACK,
                confidence=arg.confidence,
            ))

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
    generic = {d: tau for d in _WEIGHTS}
    strengths.append(ArgumentStrength(
        argument_id=arg.id,
        tau=tau,
        rubric=_make_rubric_score(generic),
        justification="Fallback: rubric not provided by LLM.",
        model="fallback",
    ))


class ArgumentExtractor:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        mcfg = cfg.get("mining", {})
        self.max_args = mcfg.get("max_arguments", 8)
        self.min_conf = mcfg.get("min_confidence", 0.3)
        self.inc_neutral = mcfg.get("include_neutral_relations", False)
        self.max_retries = mcfg.get("max_retries", 2)
        self._tmpl = _load_template()

    def _prompt(self, case_id: str, narrative: str, decision_problem: str) -> str:
        return _render(self._tmpl, case_id, narrative, self.max_args, decision_problem)

    def extract_combined(
        self,
        case_id: str,
        narrative: str,
        decision_problem: str | None = None,
    ) -> tuple[ExtractionResult, list[ArgumentStrength]]:
        decision_problem = (decision_problem or "").strip()
        if not decision_problem:
            decision_problem = _GENERIC_FALLBACK_DECISION

        logger.info(f"[{case_id}] Extractor decision problem: {decision_problem}")
        prompt = self._prompt(case_id, narrative, decision_problem)
        raw = ""
        for attempt in range(1, self.max_retries + 1):
            try:
                raw = _llm.call_llm(prompt, self.cfg)
                logger.debug(f"[{case_id}] Raw LLM response (first 400): {raw[:400]}")
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
                            case_id=case_id,
                            input_text=narrative,
                            claim=ClaimNode(text=decision_problem),
                            raw_llm_response=raw,
                            parse_error=str(exc),
                        ),
                        [],
                    )

        return (
            ExtractionResult(
                case_id=case_id,
                input_text=narrative,
                claim=ClaimNode(text=decision_problem),
                raw_llm_response=raw,
            ),
            [],
        )
