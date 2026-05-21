# ADSS-Hearsay: A Neuro-Symbolic Argumentative Decision Support System

**ADSS-Hearsay** is a research-oriented Python system for **explainable legal decision support** on the LegalBench *hearsay* task. It combines large language model (LLM) argument mining with symbolic quantitative bipolar argumentation frameworks (QBAFs) to produce transparent **Yes / No / UNCERTAIN** decisions.

The system was originally designed for deciding whether a legal narrative contains **hearsay under FRE 801(c)**, and has been extended with a more generic decision-support dashboard where a user may provide or auto-extract a custom decision problem φ.

---

## Key Contributions

- **Neuro-symbolic legal reasoning**: LLM-based extraction of arguments, argument strengths, and support/attack relations, followed by symbolic QBAF reasoning.
- **Explainable decision outputs**: Every prediction is represented as a structured argument graph with intrinsic strengths τ and final dialectical scores σ.
- **Uncertainty-aware classification**: The system returns `UNCERTAIN` when the central claim score σ(φ) lies inside a configurable uncertainty band.
- **Human-in-the-loop dashboard**: A Streamlit interface supports case input, decision-problem auto-extraction, graph visualization, and manual intervention.
- **Research-grade evaluation suite**: Implements accuracy, macro-F1, bootstrap confidence intervals, McNemar significance testing, contestability simulation, robustness perturbations, and structured error analysis.

---

## System Overview

Given an input legal narrative and a binary decision problem φ, ADSS-Hearsay performs the following pipeline:

```text
Input narrative + decision problem φ
        │
        ▼
┌─────────────────────────────┐
│ Module A: Argument Mining   │
│ LLM → arguments + relations │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Module B: Strength Scoring  │
│ LLM / fallback → τ(αᵢ)      │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Module C: QBAF Reasoning    │
│ DF-QuAD / QE → σ(φ)         │
└──────────────┬──────────────┘
               │
       ┌───────┴────────┐
       ▼                ▼
   Yes / No        UNCERTAIN
       │                │
       ▼                ▼
┌─────────────────────────────┐
│ Module D: HITL Dashboard    │
│ inspect, edit, contest      │
└─────────────────────────────┘
```

---

## Decision Semantics

The central claim is denoted as φ. For the LegalBench hearsay task, the default decision problem is:

```text
Is the statement described in this narrative hearsay under FRE 801(c)?
```

For generic use through the dashboard, φ can be manually provided or automatically extracted from the case narrative using the selected LLM backend.

The final score σ(φ) is interpreted as:

```text
Yes        if σ(φ) ≥ decision_threshold
No         if σ(φ) < decision_threshold
UNCERTAIN  if σ(φ) ∈ uncertainty_band
```

By default:

```yaml
decision_threshold: 0.5
uncertainty_band: [0.45, 0.55]
```

---

## Solvers

ADSS-Hearsay supports two QBAF semantics:

### DF-QuAD Semantics

```text
CS(x) = 1 − ∏s∈S(x)(1 − σ(s))
CA(x) = 1 − ∏a∈A(x)(1 − σ(a))
σ(x)  = clamp(τ(x) + (1 − τ(x))CS(x) − τ(x)CA(x))
```

### QE Semantics

```text
σ(x) = (τ(x) + Σσ(S(x))) / (1 + Σσ(S(x)) + Σσ(A(x)))
```

The solver can be selected in `configs/config.yaml`:

```yaml
qbaf:
  solver: "df_quad"      # or "qe_semantics"
```

---

## Repository Structure

```text
adss_hearsay/
├── configs/
│   └── config.yaml
├── prompts/
│   ├── generic_combined.txt
│   ├── combined.txt
│   ├── extraction.txt
│   ├── strength.txt
│   └── cot_baseline.txt
├── scripts/
│   ├── run_evaluation_suite.py
│   ├── run_eval.py
│   └── case_studies.py
├── src/
│   ├── dashboard/
│   │   └── app.py
│   ├── data/
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── simulations.py
│   │   ├── robustness.py
│   │   └── error_analysis.py
│   ├── mining/
│   │   └── extractor.py
│   ├── pipeline/
│   │   └── orchestrator.py
│   ├── qbaf/
│   │   ├── graph.py
│   │   └── solver.py
│   ├── scoring/
│   └── utils/
└── tests/
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/nkaranik/adss_hearsay.git
cd adss_hearsay
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

---

## LLM Backends

The system supports two LLM backends.

### Google Gemini

Set the Gemini API key:

Windows PowerShell:

```powershell
$env:GEMINI_API_KEY="your_key_here"
```

Linux/macOS:

```bash
export GEMINI_API_KEY="your_key_here"
```

Then run:

```bash
python -m streamlit run src/dashboard/app.py
```

### LM Studio

Start LM Studio locally with an OpenAI-compatible server, then select `lmstudio` in the dashboard sidebar.

Default endpoint:

```text
http://127.0.0.1:1234/v1
```

The model name in `configs/config.yaml` or the dashboard must match the model loaded in LM Studio.

---

## Running the Dashboard

```bash
python -m streamlit run src/dashboard/app.py
```

The dashboard supports:

- legal narrative input;
- manual decision-problem φ input;
- LLM-based auto-extraction of φ;
- argument extraction and scoring;
- QBAF graph visualization;
- Yes / No / UNCERTAIN decisions;
- human-in-the-loop interventions on τ values and relations.

---

## Running the Evaluation Suite

The full evaluation suite is implemented in:

```text
scripts/run_evaluation_suite.py
```

Quick smoke test:

```bash
python scripts/run_evaluation_suite.py --max 5 --skip-baselines --skip-contestability --skip-robustness
```

Small full evaluation:

```bash
python scripts/run_evaluation_suite.py --split test --max 20 --skip-robustness
```

Full evaluation:

```bash
python scripts/run_evaluation_suite.py --split test --out artifacts/evaluation_suite
```

Use Gemini explicitly:

```bash
python scripts/run_evaluation_suite.py --backend gemini --max 20
```

Use LM Studio explicitly:

```bash
python scripts/run_evaluation_suite.py --backend lmstudio --max 20
```

---

## Evaluation Protocol

The evaluation suite implements the experimental protocol required for the ADSS research setup.

### 1. Main Predictive Metrics

For the Full ADSS and baselines, the suite reports:

- **Accuracy**
- **Macro-F1**
- **95% bootstrap confidence intervals**
- **McNemar paired significance tests** against the Full ADSS

Baselines:

- Zero-shot Chain-of-Thought
- Few-shot prompting
- Non-symbolic ADSS aggregation

### 2. Uncertainty Evaluation

Predictions are partitioned into:

```text
Certain:     σ(φ) outside [0.45, 0.55]
Borderline:  σ(φ) inside  [0.45, 0.55]
```

The suite also computes:

```text
False-Certainty Rate = incorrect certain predictions / total labelled predictions
```

### 3. Automated Contestability Simulation

Because manually editing every argument graph is infeasible at evaluation scale, the suite implements two automated HITL simulation regimes.

#### Oracle-Guided Simulation

Uses the gold label to estimate an upper bound on correctability by selecting interventions likely to move the decision toward the correct class.

#### Confidence-Guided Simulation

Does not use gold labels. Instead, it targets low-confidence arguments or relations.

Metrics:

- **Decision Flip Rate (DFR)**
- **Corrective Flip Rate (CFR)**
- **Minimal Edit Count (MEC)**
- **Score Shift Magnitude (SSM)**

### 4. Robustness and Perturbation Testing

The robustness suite evaluates stability under noisy argument extraction:

1. Random argument removal
2. Random support/attack relation flipping
3. Random τ perturbation

The primary stability metric is:

```text
Δp = mean |σ_original(φ) − σ_perturbed(φ)|
```

The suite also reports perturbed accuracy and macro-F1.

### 5. Structured Error Analysis

Misclassifications are categorized into four failure modes:

1. **Argument Omission**
2. **Relation Error**
3. **Strength Miscalibration**
4. **Threshold / Uncertainty Failure**

---

## Evaluation Outputs

By default, evaluation results are written to:

```text
artifacts/evaluation_suite/
```

Expected files:

```text
artifacts/evaluation_suite/
├── main_results.json
├── main_results.csv
├── mcnemar_tests.json
├── contestability_metrics.json
├── contestability_metrics.csv
├── robustness_results.json
├── robustness_results.csv
├── error_analysis.json
└── error_cases.csv
```

These files are designed to populate tables for:

- Main Results
- Ablation Study
- Contestability Metrics
- Robustness Analysis
- Error Analysis

---

## Configuration

Main configuration lives in:

```text
configs/config.yaml
```

Important options:

```yaml
backend: "gemini"        # gemini or lmstudio

mining:
  max_arguments: 6
  min_confidence: 0.3
  include_neutral_relations: false

qbaf:
  solver: "df_quad"
  decision_threshold: 0.5
  uncertainty_band: [0.45, 0.55]
  phi_initial_strength: 0.5

evaluation:
  bootstrap_samples: 1000
  confidence_level: 0.95
  output_dir: "artifacts/evaluation_suite"
```

---

## Reproducibility

The evaluation suite uses fixed random seeds for:

- bootstrap confidence intervals;
- contestability simulations;
- robustness perturbations.

The default seed is:

```yaml
seed: 42
```

For deterministic experiments, use the same configuration, backend model, dataset split, and random seed.

---

## Testing

Run unit and integration tests:

```bash
pytest tests/ -v
```

Compile-check key files:

```bash
python -m py_compile src/evaluation/metrics.py
python -m py_compile src/evaluation/simulations.py
python -m py_compile src/evaluation/robustness.py
python -m py_compile src/evaluation/error_analysis.py
python -m py_compile scripts/run_evaluation_suite.py
```

---

## Example Use Case

A contract-like case can be analyzed through the generic dashboard by entering a custom decision problem such as:

```text
Is the plaintiff entitled to compensation?
```

or by clicking **Auto-extract** to let the selected LLM infer φ from the narrative.

For the LegalBench hearsay benchmark, use the default hearsay decision problem:

```text
Is the statement described in this narrative hearsay under FRE 801(c)?
```

---

## Limitations

- LLM argument extraction may omit relevant arguments or misclassify support/attack stance.
- QBAF scores depend on the quality of extracted arguments, relations, and τ values.
- Oracle-guided contestability simulation is an upper-bound estimate and should not be interpreted as real human performance.
- Confidence-guided intervention is a heuristic approximation of human review.
- Legal outputs are research predictions, not legal advice.

---

## Citation / Research Use

If this repository is used in an academic project, report:

- LLM backend and model version;
- dataset split and sample size;
- QBAF solver;
- uncertainty band;
- bootstrap sample count;
- random seed;
- whether baselines, contestability, and robustness modules were enabled.

---

## License

Add your preferred license here.

---

## Disclaimer

This project is intended for research and educational purposes. It does not provide legal advice and should not be used as a substitute for qualified legal judgment.
