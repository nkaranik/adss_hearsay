
"""
Evaluation metrics for ADSS-Hearsay.

Implements:
- Accuracy and Macro-F1
- 95% bootstrap confidence intervals
- McNemar paired significance test
- Certain / Borderline partitioning by sigma(phi)
- False-Certainty Rate

Designed to work with src.data.models.ADSSPrediction and BaselinePrediction,
but also accepts lightweight dict-like objects with equivalent fields.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from math import comb
from pathlib import Path
from typing import Any, Callable, Iterable
import csv
import json
import random

LABELS = ("Yes", "No")


@dataclass
class ScoreCI:
    score: float
    ci_low: float
    ci_high: float


@dataclass
class SystemMetrics:
    system_name: str
    n: int
    n_uncertain: int
    accuracy: ScoreCI
    macro_f1: ScoreCI
    false_certainty_rate: float
    certain_n: int
    certain_accuracy: float
    certain_macro_f1: float
    borderline_n: int
    borderline_accuracy: float
    borderline_macro_f1: float


@dataclass
class McNemarResult:
    system_a: str
    system_b: str
    n01: int  # A wrong, B correct
    n10: int  # A correct, B wrong
    statistic: float
    p_value: float


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def decision_value(pred: Any, uncertain_as: str = "No") -> str:
    d = _get(pred, "decision")
    if hasattr(d, "value"):
        d = d.value
    if d in ("Yes", "No"):
        return d
    return uncertain_as


def gold_value(pred: Any) -> str | None:
    g = _get(pred, "gold_label")
    if g in ("Yes", "No"):
        return g
    return None


def sigma_phi(pred: Any) -> float | None:
    s = _get(pred, "sigma_phi")
    try:
        return None if s is None else float(s)
    except Exception:
        return None


def is_uncertain(pred: Any, band: tuple[float, float] = (0.45, 0.55)) -> bool:
    d = _get(pred, "decision")
    if hasattr(d, "value"):
        d = d.value
    if d == "UNCERTAIN":
        return True
    s = sigma_phi(pred)
    return False if s is None else band[0] <= s <= band[1]


def paired_gold_pred(predictions: Iterable[Any], uncertain_as: str = "No") -> tuple[list[str], list[str]]:
    golds, preds = [], []
    for p in predictions:
        g = gold_value(p)
        if g is None:
            continue
        golds.append(g)
        preds.append(decision_value(p, uncertain_as=uncertain_as))
    return golds, preds


def accuracy_score(golds: list[str], preds: list[str]) -> float:
    return sum(g == p for g, p in zip(golds, preds)) / len(golds) if golds else 0.0


def macro_f1_score(golds: list[str], preds: list[str], labels: tuple[str, ...] = LABELS) -> float:
    if not golds:
        return 0.0
    f1s = []
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(golds, preds))
        fp = sum(g != label and p == label for g, p in zip(golds, preds))
        fn = sum(g == label and p != label for g, p in zip(golds, preds))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s)


def bootstrap_ci(
    golds: list[str],
    preds: list[str],
    metric_fn: Callable[[list[str], list[str]], float],
    n_samples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> ScoreCI:
    if not golds:
        return ScoreCI(0.0, 0.0, 0.0)
    rng = random.Random(seed)
    n = len(golds)
    values = []
    for _ in range(n_samples):
        idx = [rng.randrange(n) for _ in range(n)]
        bg = [golds[i] for i in idx]
        bp = [preds[i] for i in idx]
        values.append(metric_fn(bg, bp))
    values.sort()
    alpha = 1.0 - confidence
    lo_i = max(0, min(len(values) - 1, int((alpha / 2) * len(values))))
    hi_i = max(0, min(len(values) - 1, int((1 - alpha / 2) * len(values)) - 1))
    return ScoreCI(metric_fn(golds, preds), values[lo_i], values[hi_i])


def exact_mcnemar_p_value(n01: int, n10: int) -> float:
    """Two-sided exact binomial McNemar p-value."""
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    tail = sum(comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def mcnemar_test(
    full_predictions: Iterable[Any],
    baseline_predictions: Iterable[Any],
    system_a: str = "Full ADSS",
    system_b: str = "Baseline",
    uncertain_as: str = "No",
) -> McNemarResult:
    full_by_id = {_get(p, "case_id"): p for p in full_predictions}
    base_by_id = {_get(p, "case_id"): p for p in baseline_predictions}
    n01 = n10 = 0
    for cid in sorted(set(full_by_id) & set(base_by_id)):
        fp = full_by_id[cid]
        bp = base_by_id[cid]
        g = gold_value(fp) or gold_value(bp)
        if g is None:
            continue
        full_correct = decision_value(fp, uncertain_as) == g
        base_correct = decision_value(bp, uncertain_as) == g
        if (not full_correct) and base_correct:
            n01 += 1
        elif full_correct and (not base_correct):
            n10 += 1
    n = n01 + n10
    statistic = ((abs(n01 - n10) - 1) ** 2 / n) if n else 0.0  # continuity-corrected
    return McNemarResult(system_a, system_b, n01, n10, statistic, exact_mcnemar_p_value(n01, n10))


def partition_predictions(predictions: Iterable[Any], band: tuple[float, float] = (0.45, 0.55)) -> tuple[list[Any], list[Any]]:
    certain, borderline = [], []
    for p in predictions:
        if gold_value(p) is None:
            continue
        (borderline if is_uncertain(p, band) else certain).append(p)
    return certain, borderline


def false_certainty_rate(predictions: Iterable[Any], band: tuple[float, float] = (0.45, 0.55), uncertain_as: str = "No") -> float:
    """Incorrect predictions made outside the uncertainty band / all predictions with gold."""
    total = false_certain = 0
    for p in predictions:
        g = gold_value(p)
        if g is None:
            continue
        total += 1
        if not is_uncertain(p, band) and decision_value(p, uncertain_as) != g:
            false_certain += 1
    return false_certain / total if total else 0.0


def evaluate_system(
    predictions: list[Any],
    system_name: str,
    band: tuple[float, float] = (0.45, 0.55),
    bootstrap_samples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
    uncertain_as: str = "No",
) -> SystemMetrics:
    golds, preds = paired_gold_pred(predictions, uncertain_as=uncertain_as)
    acc_ci = bootstrap_ci(golds, preds, accuracy_score, bootstrap_samples, confidence, seed)
    f1_ci = bootstrap_ci(golds, preds, macro_f1_score, bootstrap_samples, confidence, seed + 1)
    certain, borderline = partition_predictions(predictions, band)

    cg, cp = paired_gold_pred(certain, uncertain_as=uncertain_as)
    bg, bp = paired_gold_pred(borderline, uncertain_as=uncertain_as)

    return SystemMetrics(
        system_name=system_name,
        n=len(golds),
        n_uncertain=sum(is_uncertain(p, band) for p in predictions if gold_value(p) is not None),
        accuracy=acc_ci,
        macro_f1=f1_ci,
        false_certainty_rate=false_certainty_rate(predictions, band, uncertain_as),
        certain_n=len(cg),
        certain_accuracy=accuracy_score(cg, cp),
        certain_macro_f1=macro_f1_score(cg, cp),
        borderline_n=len(bg),
        borderline_accuracy=accuracy_score(bg, bp),
        borderline_macro_f1=macro_f1_score(bg, bp),
    )


def write_metrics_report(metrics: list[SystemMetrics], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = [asdict(m) for m in metrics]
    (out / "main_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with (out / "main_results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "system_name", "n", "n_uncertain",
            "accuracy", "accuracy_ci_low", "accuracy_ci_high",
            "macro_f1", "macro_f1_ci_low", "macro_f1_ci_high",
            "false_certainty_rate", "certain_n", "certain_accuracy", "certain_macro_f1",
            "borderline_n", "borderline_accuracy", "borderline_macro_f1",
        ])
        w.writeheader()
        for m in metrics:
            w.writerow({
                "system_name": m.system_name,
                "n": m.n,
                "n_uncertain": m.n_uncertain,
                "accuracy": m.accuracy.score,
                "accuracy_ci_low": m.accuracy.ci_low,
                "accuracy_ci_high": m.accuracy.ci_high,
                "macro_f1": m.macro_f1.score,
                "macro_f1_ci_low": m.macro_f1.ci_low,
                "macro_f1_ci_high": m.macro_f1.ci_high,
                "false_certainty_rate": m.false_certainty_rate,
                "certain_n": m.certain_n,
                "certain_accuracy": m.certain_accuracy,
                "certain_macro_f1": m.certain_macro_f1,
                "borderline_n": m.borderline_n,
                "borderline_accuracy": m.borderline_accuracy,
                "borderline_macro_f1": m.borderline_macro_f1,
            })
