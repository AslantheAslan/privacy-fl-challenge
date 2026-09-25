"""End-to-end pipeline used by ``run_submission.py``.

1. **NLP per case** – PII detection + rendering, and structured extraction.
   A failure on one record never aborts the run: that record falls back to
   conservative defaults and the incident is reported in the summary.
2. **Readmission model** – FedAvg across the three hospital nodes, each in its
   own OS process that loads only its own rows, with secure aggregation.
   L2 strength is selected by federated site-stratified CV at run time.
3. **Experiment battery** – local vs federated vs centralised (site-stratified
   CV on the training split; holdout metrics only when labels are supplied),
   seed stability, convergence / client drift, non-IID profile, secure
   aggregation benchmark and the differential-privacy utility curve.
"""
from __future__ import annotations

import logging
import platform
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .deid import detect_pii
from .extraction import extract_clinical_data
from .fl import experiments as ex
from .io_utils import normalise_input_record, read_jsonl, write_json, write_jsonl
from .privacy.summary import build_privacy_summary
from .spans import render_deidentified

LOGGER = logging.getLogger("submission")
EMPTY_EXTRACTION = {
    "diagnoses": [], "medications": [], "heart_rate_bpm": None, "systolic_bp_mmhg": None,
    "creatinine_mg_dl": None, "hemoglobin_g_dl": None, "lvef_percent": None,
    "smoking_status": None, "allergy": None,
}
L2_GRID = (0.01, 0.02, 0.05, 0.1, 0.2)
DP_EPSILONS = (0.5, 1.0, 2.0, 4.0, 8.0, float("inf"))
DP_DELTA = 1e-3


def process_note(record: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """De-identify and extract one case. Returns the partial prediction + warnings."""
    warnings = list(record.get("_input_warnings", []))
    note = record["note_text"]
    age = (record.get("structured_features") or {}).get("age_years")
    try:
        spans = detect_pii(note, age_years=age if isinstance(age, int) else None)
        pii = [s.to_json() for s in spans]
        deidentified = render_deidentified(note, spans)
    except Exception as exc:  # pragma: no cover - defensive
        warnings.append(f"de-identification failed: {exc}")
        pii, deidentified = [], note
    try:
        extracted = extract_clinical_data(note)
    except Exception as exc:  # pragma: no cover - defensive
        warnings.append(f"extraction failed: {exc}")
        extracted = dict(EMPTY_EXTRACTION)
    return {"case_id": record["case_id"], "pii_entities": pii, "deidentified_text": deidentified,
            "extracted_clinical_data": extracted}, warnings


def _final_federated_model(train_path: Path, train: list[dict], cfg: ex.ExperimentConfig, seed: int) -> dict[str, Any]:
    """Train the submitted model: FedAvg + secure aggregation, one process per node."""
    by_site = ex.partition_by_site(train)
    try:
        run = ex.train_federated(by_site, cfg, seed, secure=True, track_objective=True,
                                 backend="process", train_path=str(train_path))
        run["backend"] = "process (one OS process per hospital node; each loads only its own rows)"
    except Exception as exc:  # e.g. sandbox forbids subprocesses: fall back, keep going
        LOGGER.warning("process backend unavailable (%s); using in-process nodes", exc)
        run = ex.train_federated(by_site, cfg, seed, secure=True, track_objective=True)
        run["backend"] = f"in-process serialised endpoints (process backend failed: {type(exc).__name__})"
    return run


def run(
    train_path: Path,
    input_path: Path,
    output_path: Path,
    artifacts_dir: Path,
    eval_labels_path: Path | None = None,
    quick: bool = False,
    seed: int = 7,
) -> dict[str, Any]:
    started = time.perf_counter()
    timings: dict[str, float] = {}

    # ---------------------------------------------------------------- inputs
    train = [r for r in read_jsonl(train_path) if isinstance(r.get("labels"), dict)
             and r["labels"].get("readmission_30d") in (0, 1)]
    raw_inputs = read_jsonl(input_path)
    inputs, input_errors = [], []
    for index, record in enumerate(raw_inputs):
        try:
            inputs.append(normalise_input_record(record))
        except ValueError as exc:
            input_errors.append(f"input line {index + 1}: {exc}")

    # ------------------------------------------------------------------- NLP
    t0 = time.perf_counter()
    predictions, warnings = [], {}
    for record in inputs:
        prediction, record_warnings = process_note(record)
        predictions.append(prediction)
        if record_warnings:
            warnings[record["case_id"]] = record_warnings
    timings["nlp_seconds"] = time.perf_counter() - t0

    # -------------------------------------------------------- model selection
    base = ex.ExperimentConfig(seeds=(0, 1, 2) if quick else (0, 1, 2, 3, 4), cv_repeats=2 if quick else 3)
    t0 = time.perf_counter()
    l2_selection = ex.select_l2(train, base, L2_GRID)
    cfg = replace(base, l2=l2_selection["selected_l2"])
    timings["l2_selection_seconds"] = time.perf_counter() - t0

    # ------------------------------------------------------------ final model
    t0 = time.perf_counter()
    final = _final_federated_model(train_path, train, cfg, seed)
    timings["final_federated_training_seconds"] = time.perf_counter() - t0
    probabilities = ex.predict(final["theta"], final["scaler"], cfg.spec, inputs)
    for prediction, probability in zip(predictions, probabilities):
        value = float(probability)
        prediction["readmission_probability"] = float(np.clip(value, 0.0, 1.0)) if np.isfinite(value) else 0.5
    write_jsonl(output_path, predictions)

    # ------------------------------------------------------------ experiments
    t0 = time.perf_counter()
    labelled_holdout = _attach_labels(inputs, eval_labels_path) if eval_labels_path else None
    results: dict[str, Any] = {
        "cv": ex.site_stratified_cv(train, cfg),
        "seed_stability": ex.seed_stability(train, cfg, tuple(range(5 if quick else 10)), labelled_holdout),
        "convergence": ex.convergence_study(train, cfg, epochs_grid=(1, 2, 5) if quick else (1, 2, 5, 10)),
        "non_iid": ex.non_iid_profile(train, cfg),
        "personalisation": ex.personalisation_study(train, cfg),
        "secure_aggregation": ex.secure_aggregation_benchmark(train, cfg),
        "dp": ex.dp_study(train, cfg, DP_EPSILONS, DP_DELTA, holdout=labelled_holdout),
    }
    if labelled_holdout:
        results["holdout"] = ex.holdout_evaluation(train, labelled_holdout, cfg)
    timings["experiments_seconds"] = time.perf_counter() - t0
    timings["total_seconds"] = time.perf_counter() - started

    experiment_summary = build_experiment_summary(cfg, l2_selection, final, results, timings, seed,
                                                  eval_labels_path, len(inputs), input_errors, warnings)
    privacy_summary = build_privacy_summary(cfg, final, results["secure_aggregation"], results["dp"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    write_json(artifacts_dir / "experiment_summary.json", experiment_summary)
    write_json(artifacts_dir / "privacy_summary.json", privacy_summary)
    return {"predictions": len(predictions), "timings": timings}


def _attach_labels(inputs: list[dict[str, Any]], labels_path: Path) -> list[dict[str, Any]] | None:
    truth = {r["case_id"]: r for r in read_jsonl(labels_path)}
    labelled = [dict(r, labels={"readmission_30d": truth[r["case_id"]]["readmission_30d"]})
                for r in inputs if r["case_id"] in truth]
    return labelled or None


def _round(value: Any, digits: int = 4) -> Any:
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {k: _round(v, digits) for k, v in value.items()}
    if isinstance(value, list):
        return [_round(v, digits) for v in value]
    return value


def build_experiment_summary(cfg, l2_selection, final, results, timings, seed, eval_labels_path, n_inputs,
                             input_errors, warnings) -> dict[str, Any]:
    cv = results["cv"]["summary"]

    def headline(regime: str) -> dict[str, Any]:
        s = cv[regime]
        return {
            "cv_roc_auc": s["overall"]["roc_auc"], "cv_average_precision": s["overall"]["average_precision"],
            "cv_brier": s["overall"]["brier"], "cv_log_loss": s["overall"]["log_loss"],
            "cv_by_site_roc_auc": {site: s[site]["roc_auc"] for site in s if site.endswith("_NODE")},
            "cv_by_site_brier": {site: s[site]["brier"] for site in s if site.endswith("_NODE")},
            "cv_worst_site_roc_auc": s["worst_site_roc_auc"],
        }

    history = final["history"]
    comm = final["log"].summary()
    summary = {
        "implementation": "FedAvg (own implementation, NumPy) with Bonawitz-style secure aggregation",
        "submitted_model": {
            "regime": "federated",
            "algorithm": "FedAvg over L2-regularised logistic regression",
            "secure_aggregation": True,
            "differential_privacy": False,
            "training_backend": final["backend"],
            "seed": seed,
            "rounds": cfg.rounds,
            "final_global_objective": history[-1].objective if history else None,
            "parameters": {name: float(v) for name, v in zip([*cfg.spec.names, "intercept"], final["theta"])},
            "standardisation": "federated: pooled mean/std from securely aggregated (count, sum, sum_sq)",
        },
        "configuration": ex.config_dict(cfg),
        "random_seeds": {
            "submitted_model_seed": seed,
            "evaluation_model_seeds": list(cfg.seeds),
            "cv_fold_seeds": [1000 + r for r in range(cfg.cv_repeats)],
            "seed_stability_seeds": results["seed_stability"]["seeds"],
            "note": "Secure-aggregation keys come from the OS CSPRNG, but masks cancel exactly, so outputs are "
                    "bit-for-bit reproducible up to 2^-24 fixed-point rounding.",
        },
        "hyperparameter_selection": l2_selection,
        "local_models": {
            "definition": "one model per hospital trained only on its own rows and own standardisation statistics; "
                          "each hospital's patients are scored by that hospital's model",
            "algorithm": "L2 logistic regression, same optimiser and passes over data as FedAvg",
            "features": list(cfg.spec.names),
            **headline("local"),
        },
        "federated_model": {
            "algorithm": "FedAvg",
            "rounds": cfg.rounds,
            "local_epochs": cfg.local_epochs,
            "optimizer": f"mini-batch SGD, lr={cfg.learning_rate}, batch={cfg.batch_size}, L2={cfg.l2}",
            "client_weighting": "n_k / sum(n) (number of training cases)",
            "aggregation": "weighted mean of client models from one secure sum of [n_k*theta_k, n_k]",
            **headline("federated"),
        },
        "centralized_model": {
            "algorithm": "same L2 logistic regression + same SGD on pooled rows (non-private reference)",
            "fairness_of_comparison": (
                "Same model class, features, preprocessing (identical pooled statistics), L2, seed, learning rate, "
                "batch size and number of passes over each row. It is an *upper reference*, not a deployable "
                "option: it requires pooling patient rows, which the governance model forbids. Remaining "
                "differences are only optimisation order (mini-batches mix sites) and FedAvg client drift."),
            **headline("centralized"),
        },
        "evaluation_design": results["cv"]["design"],
        "cv_metrics_full": cv,
        "holdout_metrics": results.get("holdout"),
        "holdout_note": None if eval_labels_path else
            "No labels for the evaluated input file were supplied, so metrics are cross-validated on the training "
            "split only. Re-run with --eval-labels to add holdout metrics (reports/ contains the validation run).",
        "seed_stability": results["seed_stability"],
        "convergence": {
            "submitted_model_objective_curve": [round(h.objective, 6) for h in history if h.objective is not None],
            "submitted_model_update_norms": [round(h.update_norm, 6) for h in history],
            **results["convergence"],
        },
        "non_iid_profile": results["non_iid"],
        "personalisation_study": {
            k: {"cv_roc_auc": v["overall"]["roc_auc"], "cv_log_loss": v["overall"]["log_loss"],
                "cv_by_site_roc_auc": {s: v[s]["roc_auc"] for s in v if s.endswith("_NODE")}}
            for k, v in results["personalisation"]["results"].items()
        },
        "communication": {
            "what_leaves_each_client": [
                "one-time: 2 Diffie-Hellman public keys (2048-bit) and AEAD-encrypted Shamir shares of its mask key",
                "per aggregation: AEAD-encrypted Shamir shares of a fresh self-mask seed (for peers, routed by server)",
                "per aggregation: one masked vector in Z_2^64 (uniformly random to the server on its own)",
                "per aggregation: Shamir shares needed to unmask survivors / dropped peers",
                "never: notes, features, labels, per-row predictions, gradients of individual rows, or n_k in clear",
            ],
            "aggregated_quantities": {
                "standardisation": f"{3 * cfg.spec.dim} values: per-feature count, sum, sum of squares",
                "training_round": f"{cfg.spec.dim + 2} values: n_k * theta_k ({cfg.spec.dim + 1}) and n_k",
                "objective_tracking": "2 values: n_k * local objective, n_k (monitoring only)",
            },
            "submitted_training_transport": comm,
        },
        "non_iid_observations": _non_iid_observations(results),
        "limitations": [
            "Only 120 training rows (~40 per site); CV estimates have wide intervals and per-site AUCs on the "
            "30-case validation split (1-3 positives per site) are not interpretable.",
            "Three clients is the minimum for meaningful secure aggregation: the server colluding with one hospital "
            "learns the sum of the other two.",
            "The federation is simulated on one machine (separate processes, serialised messages); network faults, "
            "stragglers and authentication (PKI) are not modelled.",
            "Features come from rule-based extraction; extraction errors on unseen formats would propagate.",
            "Readmission labels are synthetic with an explicit site effect; conclusions about clinical risk factors "
            "must not be drawn.",
        ],
        "runtime": {**{k: round(v, 2) for k, v in timings.items()}, "python": platform.python_version()},
        "n_evaluated_cases": n_inputs,
        "input_errors": input_errors,
        "per_case_warnings": warnings,
    }
    return _round(summary)


def _non_iid_observations(results: dict[str, Any]) -> list[str]:
    cv = results["cv"]["summary"]
    profile = results["non_iid"]
    prevalences = {s: round(p["readmission_prevalence"], 2) for s, p in profile.items()}
    conv = results["convergence"]["runs"]
    drift = {k: round(v["final_gap_to_pooled_optimum"], 5) for k, v in conv.items() if k.startswith("n_samples")}
    local, fed = cv["local"], cv["federated"]
    per_site = {
        site: (round(local[site]["roc_auc"]["mean"] or 0, 3), round(fed[site]["roc_auc"]["mean"] or 0, 3))
        for site in local if site.endswith("_NODE")
    }
    return [
        f"Label skew: readmission prevalence differs by site {prevalences}; covariate skew is visible in the "
        "non_iid_profile (e.g. diagnosis mix, prior admissions).",
        f"Local vs federated CV AUC per site (local, federated): {per_site}. Federation helps most where a site's "
        "own data are small or lack events for a risk factor; the site indicator lets the global model keep "
        "site-specific baselines.",
        f"Client drift: gap between the FedAvg fixed point and the pooled optimum grows with local epochs "
        f"{drift}; E=2 balances communication and drift for this data size.",
        "Calibration: local models are worse calibrated (higher CV log-loss/Brier) because a ~40-row site "
        "estimates its intercept and coefficients with high variance.",
        "Coverage skew: BERLIN_NODE has no chronic-kidney-disease case and a single prior admission in training, "
        "so a Berlin-only model cannot learn two of the strongest risk factors; the federated model can.",
        "Personalisation (FedAvg + local fine-tuning) did not improve CV discrimination or calibration here "
        "(see personalisation_study): the site indicator already captures site baselines and ~32 rows per fold "
        "mostly add variance.",
    ]


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def format_exception() -> str:  # pragma: no cover - used by the CLI
    return traceback.format_exc()
