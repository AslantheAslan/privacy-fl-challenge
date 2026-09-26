# Privacy-Preserving Clinical AI & Cross-Silo Federated Learning

Solution to the synthetic take-home challenge (brief: [`docs/CHALLENGE_README.md`](docs/CHALLENGE_README.md)).
Three hospital nodes — `BERLIN_NODE`, `CHENNAI_NODE`, `HYDERABAD_NODE` — de-identify and structure their
clinical notes locally, then train a 30-day readmission model together with **FedAvg over secure aggregation**,
without any patient row leaving its node.

| Public validation (official evaluator) | Score |
|---|---|
| De-identification | **1.000** (15.00 / 15) |
| Structured extraction | **1.000** (15.00 / 15) |
| Readmission prediction | 0.744 (7.44 / 10) |
| **Automated total** | **37.44 / 40** (starter: 31.84) |

Readmission, site-stratified 5-fold CV × 3 on the training split (the 6-positive validation split is too small
to rank models): **federated AUC 0.842 / Brier 0.147 ≈ centralized 0.837 / 0.147 > local 0.809 / 0.170**
(starter 0.696 / 0.218). Full analysis: [`REPORT.md`](REPORT.md).

## Quick start

```bash
python -m pip install -r requirements.txt          # numpy, scikit-learn, pytest (pinned)

python run_submission.py \
  --train data/train.jsonl \
  --input data/validation_inputs.jsonl \
  --output outputs/validation_predictions.jsonl \
  --artifacts-dir outputs/artifacts                # ≈11 s on 2 CPUs, no network needed

make evaluate     # predict + official evaluator report in outputs/
make test         # 156 tests, ≈13 s
make lint         # ruff
make report       # regenerate reports/ (adds holdout metrics via --eval-labels)
make stress       # de-identification under 10 synthetic formatting shifts
```

On Windows, without the make command:

```bash
# make evaluate:
python evaluator/evaluate.py --inputs data/validation_inputs.jsonl --ground-truth data/validation_ground_truth.jsonl --predictions outputs/validation_predictions.jsonl --report outputs/validation_report.json

# make test:
python -m pytest

# make lint:
python -m pip install ruff
python -m ruff check .

# make report:
python run_submission.py --train data/train.jsonl --input data/validation_inputs.jsonl --output reports/validation_run/validation_predictions.jsonl --artifacts-dir reports/validation_run --eval-labels data/validation_ground_truth.jsonl
python evaluator/evaluate.py --inputs data/validation_inputs.jsonl --ground-truth data/validation_ground_truth.jsonl --predictions reports/validation_run/validation_predictions.jsonl --report reports/validation_run/validation_report.json
python scripts/format_shift_eval.py --output reports/format_shift_deid.json

# make stress:
python scripts/format_shift_eval.py --verbose 

```

Optional flags: `--eval-labels <ground_truth.jsonl>` adds holdout metrics to `experiment_summary.json`;
`--quick` shortens the analysis battery (predictions are unchanged); `--seed` sets the submitted model's seed.

### Docker

```bash
docker build -t privacy-fl-challenge .
docker run --rm --network none -v "$PWD/outputs:/app/outputs" privacy-fl-challenge \
  --train data/train.jsonl --input data/validation_inputs.jsonl \
  --output outputs/validation_predictions.jsonl --artifacts-dir outputs/artifacts
```

The image (`python:3.11-slim` + NumPy and scikit-learn, expected well under 1 GB) runs a subset of the tests
during the build, runs as a non-root user and needs no network at runtime. Mount a different input file to
score the hidden set. It could not be built in the development sandbox (no Docker daemon); a clean virtualenv
built from `requirements.txt` reproduced the outputs byte-for-byte.

## What the command produces

| File | Content |
|---|---|
| `--output` | one JSON record per input case: PII spans, de-identified text, canonical extraction, readmission probability |
| `experiment_summary.json` | local / federated / centralized metrics (overall and per site), FedAvg configuration and seeds, L2 selection, convergence and client-drift study, seed stability, non-IID profile, personalisation study, exact communication log, limitations |
| `privacy_summary.json` | mechanism, protected asset, adversary, trust assumptions, cryptographic and DP parameters, exact privacy claim and non-guarantees, measured utility/runtime/bytes, dropout test, attack surface |

## How it works

* **De-identification** (`src/deid.py`) — prioritised, cue-aware rule passes (e-mail, phone, IDs, dates typed
  as DOB/encounter by the nearest cue, postal-code-anchored addresses, patient/clinician names incl. names
  recovered from e-mail addresses) with a non-overlap resolver. Robust to ten synthetic format shifts
  (`src/format_shift.py`, `reports/format_shift_deid.json`).
* **Extraction** (`src/extraction.py`) — lexicons with abbreviations and brands, NegEx-style negation /
  family-history / allergy scoping, unit conversion (µmol/L, g/L, mmol/L), decimal commas, explicit `null`s.
* **Federated learning** (`src/fl/`) — each `HospitalClient` owns its rows (in the submitted run: its own OS
  process that reads only its own site from the file) and answers only allow-listed aggregate requests over a
  pickle-free, audit-logged transport. FedAvg (`n_k`-weighted) over L2 logistic regression; standardisation from
  federated moments; L2 chosen by federated CV.
* **Privacy** (`src/privacy/`) — secure aggregation (Bonawitz et al. 2017, semi-honest, t = 2 of 3, dropout
  recovery) built on stdlib cryptography (RFC 3526 DH, SHAKE-256, HMAC-SHA256, Shamir); record-level DP-GD with
  exact μ-GDP accounting, evaluated from ε = 0.5 to 8 and off for the submitted model.

## Repository layout

```
run_submission.py        standard entry point
src/
  deid.py spans.py       PII detection + rendering
  extraction.py          canonical clinical extraction
  features.py            feature sets, federated-ready standardisation
  format_shift.py        span-aware formatting perturbations (robustness tests)
  pipeline.py            end-to-end orchestration and experiment summary
  fl/                    logreg, client, server (FedAvg), transport, metrics, experiments
  privacy/               crypto primitives, secure aggregation, DP, privacy summary
scripts/                 component_report.py (per-field errors), format_shift_eval.py
tests/                   156 tests (unit, protocol, FL invariants, end-to-end schema)
reports/                 committed validation run and format-shift results
evaluator/ schemas/ data/  challenge material (unchanged)
docs/CHALLENGE_README.md original challenge brief
```

`src/baseline.py` is the untouched starter, kept only because `tests/test_smoke.py` imports it.

## Data and licensing

All data are synthetic and were supplied with the challenge. No external data, model weights or credentials
are used or committed.
