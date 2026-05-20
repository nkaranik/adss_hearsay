"""
Evaluation module.
Reports accuracy, macro-F1, per-class metrics, confusion matrix,
and 95% bootstrap confidence intervals.
"""
from __future__ import annotations

import csv
import json
import logging
import random
from pathlib import Path
from typing import Optional, Union

from src.data.models import (
    ADSSPrediction, BaselinePrediction, ClassMetrics,
    Decision, EvaluationReport,
)

logger = logging.getLogger(__name__)
LABELS = ["Yes", "No"]


def _to_binary(d: Decision) -> str:
    return d.value if d != Decision.UNCERTAIN else "No"


def _accuracy(golds: list[str], preds: list[str]) -> float:
    return sum(g == p for g, p in zip(golds, preds)) / len(golds) if golds else 0.0


def _per_class(golds: list[str], preds: list[str]) -> list[ClassMetrics]:
    results = []
    for label in LABELS:
        tp = sum(g == label and p == label for g, p in zip(golds, preds))
        fp = sum(g != label and p == label for g, p in zip(golds, preds))
        fn = sum(g == label and p != label for g, p in zip(golds, preds))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        results.append(ClassMetrics(
            label=label, precision=prec, recall=rec,
            f1=f1, support=sum(g == label for g in golds),
        ))
    return results


def _macro_f1(pc: list[ClassMetrics]) -> float:
    return sum(m.f1 for m in pc) / len(pc) if pc else 0.0


def _conf_matrix(golds: list[str], preds: list[str]) -> list[list[int]]:
    idx = {l: i for i, l in enumerate(LABELS)}
    n   = len(LABELS)
    cm  = [[0] * n for _ in range(n)]
    for g, p in zip(golds, preds):
        if g in idx and p in idx:
            cm[idx[g]][idx[p]] += 1
    return cm


def _bootstrap_ci(
    golds: list[str],
    preds: list[str],
    metric_fn,
    n: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    rng    = random.Random(seed)
    size   = len(golds)
    scores = []
    for _ in range(n):
        idx   = [rng.randint(0, size - 1) for _ in range(size)]
        g     = [golds[i] for i in idx]
        p     = [preds[i]  for i in idx]
        scores.append(metric_fn(g, p))
    scores.sort()
    alpha = 1 - ci
    return scores[int(alpha / 2 * n)], scores[int((1 - alpha / 2) * n)]


def evaluate_predictions(
    predictions: list[Union[ADSSPrediction, BaselinePrediction]],
    system_name: str,
    cfg: dict,
    output_dir: Optional[Union[str, Path]] = None,
) -> EvaluationReport:
    golds, preds, n_unc = [], [], 0
    for p in predictions:
        if p.gold_label is None:
            continue
        dec = p.decision
        if dec == Decision.UNCERTAIN:
            n_unc += 1
        golds.append(p.gold_label)
        preds.append(_to_binary(dec))

    if not golds:
        raise ValueError("No labelled examples to evaluate.")

    n_boot = cfg.get("evaluation", {}).get("bootstrap_samples", 1000)
    conf   = cfg.get("evaluation", {}).get("confidence_level", 0.95)

    pc     = _per_class(golds, preds)
    report = EvaluationReport(
        system_name=system_name,
        accuracy=_accuracy(golds, preds),
        macro_f1=_macro_f1(pc),
        per_class=pc,
        confusion_matrix=_conf_matrix(golds, preds),
        confusion_labels=LABELS,
        accuracy_ci=_bootstrap_ci(golds, preds, _accuracy, n_boot, conf),
        macro_f1_ci=_bootstrap_ci(
            golds, preds,
            lambda g, p: _macro_f1(_per_class(g, p)),
            n_boot, conf,
        ),
        n_samples=len(golds),
        n_uncertain_escalated=n_unc,
    )

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{system_name}_report.json").write_text(
            json.dumps(report.to_dict(), indent=2), encoding="utf-8"
        )
        with open(out / f"{system_name}_predictions.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["case_id", "gold", "prediction", "sigma_phi", "uncertain"])
            for p in predictions:
                sigma = p.sigma_phi if isinstance(p, (ADSSPrediction, BaselinePrediction)) else ""
                unc   = p.uncertainty.is_uncertain if isinstance(p, ADSSPrediction) else False
                w.writerow([p.case_id, p.gold_label, _to_binary(p.decision),
                            f"{sigma:.4f}" if sigma else "", unc])

    logger.info(
        f"[{system_name}] Acc={report.accuracy:.3f} "
        f"CI95=[{report.accuracy_ci[0]:.3f},{report.accuracy_ci[1]:.3f}] "
        f"| Macro-F1={report.macro_f1:.3f} "
        f"CI95=[{report.macro_f1_ci[0]:.3f},{report.macro_f1_ci[1]:.3f}]"
    )
    return report


def print_report(r: EvaluationReport) -> None:
    print(f"\n{'='*60}")
    print(f"  {r.system_name}  (n={r.n_samples}, uncertain={r.n_uncertain_escalated})")
    print(f"  Accuracy : {r.accuracy:.3f}  [{r.accuracy_ci[0]:.3f}–{r.accuracy_ci[1]:.3f}]")
    print(f"  Macro-F1 : {r.macro_f1:.3f}  [{r.macro_f1_ci[0]:.3f}–{r.macro_f1_ci[1]:.3f}]")
    print("  Per-class:")
    for m in r.per_class:
        print(f"    {m.label:4s}  P={m.precision:.3f}  R={m.recall:.3f}  F1={m.f1:.3f}  n={m.support}")
    print(f"  Confusion matrix {r.confusion_labels}:")
    for row in r.confusion_matrix:
        print(f"    {row}")
    print("="*60)
