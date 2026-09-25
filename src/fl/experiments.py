"""Local vs federated vs centralised readmission experiments.

All three regimes share *everything* except the data they can see:
the model class (L2 logistic regression), feature set, optimiser (mini-batch
SGD, same learning rate, batch size and total number of passes over each
row = rounds x local_epochs), L2 strength and initialisation seed. The
federated and centralised models also share identical preprocessing, because
federated standardisation reproduces the pooled mean/std exactly from
aggregated moments. Local models can only use their own site's statistics.

Evaluation designs
------------------
* ``site_stratified_cv`` – K-fold CV run *within each site*, so that every node
  holds a training part and a test part of its own data, exactly like a real
  cross-silo deployment. Repeated with different fold seeds; out-of-fold
  predictions give overall and per-site metrics.
* ``holdout_evaluation`` – train on the full training split and score the
  public validation split (only 6 positives, so reported with bootstrap CIs
  and treated as a sanity check rather than a model-selection signal).
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from functools import partial
from typing import Any

import numpy as np

from ..features import FeatureSpec, Standardizer
from ..io_utils import HOSPITALS
from ..privacy.dp import DPConfig, epsilon_exact, epsilon_rdp, noise_for_epsilon
from .client import ClientConfig, HospitalClient, build_client, site_index
from .logreg import SGDConfig, fit_newton, local_sgd, objective, predict_proba
from .metrics import binary_metrics, bootstrap_auc_ci, metrics_by_site, summarize
from .server import FederatedServer
from .transport import InProcessEndpoint, ProcessEndpoint, TransportLog

Record = dict[str, Any]


@dataclass(frozen=True)
class ExperimentConfig:
    feature_set: str = "compact"
    rounds: int = 60
    local_epochs: int = 2
    learning_rate: float = 0.2
    batch_size: int = 16
    l2: float = 0.05
    weighting: str = "n_samples"
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    cv_folds: int = 5
    cv_repeats: int = 3

    @property
    def sgd(self) -> SGDConfig:
        return SGDConfig(learning_rate=self.learning_rate, batch_size=self.batch_size, l2=self.l2)

    @property
    def spec(self) -> FeatureSpec:
        return FeatureSpec.named(self.feature_set)


def label(record: Record) -> float:
    return float(record["labels"]["readmission_30d"])


def partition_by_site(records: list[Record]) -> dict[str, list[Record]]:
    parts: dict[str, list[Record]] = {site: [] for site in HOSPITALS}
    for record in records:
        parts.setdefault(record["hospital_id"], []).append(record)
    return {site: rows for site, rows in parts.items() if rows}


def predict(theta: np.ndarray, scaler: Standardizer, spec: FeatureSpec, records: list[Record]) -> np.ndarray:
    if not records:
        return np.zeros(0)
    return predict_proba(theta, scaler.transform(spec.matrix(records)))


# ---------------------------------------------------------------------------
# Training regimes
# ---------------------------------------------------------------------------
def build_federation(
    by_site: dict[str, list[Record]],
    cfg: ExperimentConfig,
    seed: int,
    secure: bool = False,
    backend: str = "inprocess",
    train_path: str | None = None,
    log: TransportLog | None = None,
) -> tuple[FederatedServer, TransportLog]:
    log = log or TransportLog()
    endpoints = {}
    for site, rows in by_site.items():
        config = ClientConfig(site=site, feature_set=cfg.feature_set, seed=seed, secure=secure,
                              train_path=train_path, row_ids=None if backend == "process" else tuple(r["case_id"] for r in rows))
        if backend == "process":
            if train_path is None:
                raise ValueError("the process backend loads data from train_path inside each node")
            endpoint = ProcessEndpoint(site, partial(build_client, config), log)
        else:
            endpoint = InProcessEndpoint(site, HospitalClient(config, rows).handle, log)
        endpoints[site_index(site)] = endpoint
    return FederatedServer(endpoints, secure=secure), log


def train_federated(
    by_site: dict[str, list[Record]],
    cfg: ExperimentConfig,
    seed: int,
    secure: bool = False,
    track_objective: bool = False,
    dp: DPConfig | None = None,
    backend: str = "inprocess",
    train_path: str | None = None,
    sgd: SGDConfig | None = None,
    dropout_schedule: dict[int, frozenset[int]] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    server, log = build_federation(by_site, cfg, seed, secure, backend, train_path)
    try:
        result = server.fit(cfg.spec.dim, cfg.rounds, cfg.local_epochs, sgd or cfg.sgd, seed, cfg.weighting, dp,
                            track_objective, dropout_schedule)
    finally:
        server.close()
    return {"theta": result.theta, "scaler": result.scaler, "history": result.history, "log": log,
            "seconds": time.perf_counter() - started}


def train_local(rows: list[Record], cfg: ExperimentConfig, seed: int) -> tuple[np.ndarray, Standardizer]:
    client = HospitalClient(ClientConfig(site=rows[0]["hospital_id"], feature_set=cfg.feature_set, seed=seed), rows)
    init = np.random.default_rng(seed).normal(0.0, 0.01, size=cfg.spec.dim + 1)
    return client.fit_local(cfg.rounds * cfg.local_epochs, cfg.sgd, init)


def train_centralized(records: list[Record], cfg: ExperimentConfig, seed: int) -> tuple[np.ndarray, Standardizer]:
    """Pooled reference: same optimiser and number of passes as FedAvg."""
    spec = cfg.spec
    x_raw = spec.matrix(records)
    scaler = Standardizer.from_flat_moments(Standardizer.local_moments(x_raw), spec.dim)
    y = np.array([label(r) for r in records])
    init = np.random.default_rng(seed).normal(0.0, 0.01, size=spec.dim + 1)
    rng = np.random.default_rng([seed, 99])
    return local_sgd(init, scaler.transform(x_raw), y, cfg.rounds * cfg.local_epochs, cfg.sgd, rng), scaler


def centralized_optimum(records: list[Record], cfg: ExperimentConfig) -> tuple[np.ndarray, Standardizer, float]:
    spec = cfg.spec
    x_raw = spec.matrix(records)
    scaler = Standardizer.from_flat_moments(Standardizer.local_moments(x_raw), spec.dim)
    x, y = scaler.transform(x_raw), np.array([label(r) for r in records])
    theta = fit_newton(x, y, cfg.l2)
    return theta, scaler, objective(theta, x, y, cfg.l2)


# ---------------------------------------------------------------------------
# Evaluation designs
# ---------------------------------------------------------------------------
def _stratified_site_folds(records: list[Record], k: int, seed: int) -> list[int]:
    """Assign folds within each site, stratified by label."""
    folds = [0] * len(records)
    rng = np.random.default_rng(seed)
    groups: dict[tuple[str, float], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[(record["hospital_id"], label(record))].append(index)
    offset = 0
    for key in sorted(groups):
        members = list(rng.permutation(groups[key]))
        for position, index in enumerate(members):
            folds[index] = (position + offset) % k
        offset += len(members)
    return folds


REGIMES = ("local", "federated", "centralized")


def _fit_all(train: list[Record], cfg: ExperimentConfig, seed: int) -> dict[str, Any]:
    by_site = partition_by_site(train)
    fed = train_federated(by_site, cfg, seed)
    central = train_centralized(train, cfg, seed)
    local = {site: train_local(rows, cfg, seed) for site, rows in by_site.items()}
    return {"federated": (fed["theta"], fed["scaler"]), "centralized": central, "local": local}


def _predict_regime(models: dict[str, Any], regime: str, cfg: ExperimentConfig, rows: list[Record]) -> np.ndarray:
    if regime != "local":
        theta, scaler = models[regime]
        return predict(theta, scaler, cfg.spec, rows)
    out = np.zeros(len(rows))
    for index, row in enumerate(rows):  # each hospital deploys its own local model
        theta, scaler = models["local"][row["hospital_id"]]
        out[index] = predict(theta, scaler, cfg.spec, [row])[0]
    return out


def site_stratified_cv(train: list[Record], cfg: ExperimentConfig) -> dict[str, Any]:
    per_repeat: dict[str, list[dict[str, Any]]] = {regime: [] for regime in REGIMES}
    y = np.array([label(r) for r in train])
    sites = [r["hospital_id"] for r in train]
    for repeat in range(cfg.cv_repeats):
        folds = np.array(_stratified_site_folds(train, cfg.cv_folds, seed=1000 + repeat))
        oof = {regime: np.zeros(len(train)) for regime in REGIMES}
        for fold in range(cfg.cv_folds):
            tr = [r for r, f in zip(train, folds) if f != fold]
            te_idx = np.where(folds == fold)[0]
            models = _fit_all(tr, cfg, seed=repeat)
            te = [train[i] for i in te_idx]
            for regime in REGIMES:
                oof[regime][te_idx] = _predict_regime(models, regime, cfg, te)
        for regime in REGIMES:
            per_repeat[regime].append(metrics_by_site(y, oof[regime], sites))
    return {"design": f"{cfg.cv_folds}-fold CV within each site, stratified by label, {cfg.cv_repeats} repeats",
            "summary": {regime: _summarize_metric_list(per_repeat[regime]) for regime in REGIMES}}


def _summarize_metric_list(items: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for scope in items[0]:
        if scope == "worst_site_roc_auc":
            out[scope] = summarize([m[scope] for m in items])
            continue
        out[scope] = {metric: summarize([m[scope][metric] for m in items])
                      for metric in ("roc_auc", "average_precision", "brier", "log_loss", "mean_predicted", "prevalence")}
    return out


def holdout_evaluation(train: list[Record], holdout: list[Record], cfg: ExperimentConfig) -> dict[str, Any]:
    y = np.array([label(r) for r in holdout])
    sites = [r["hospital_id"] for r in holdout]
    per_seed: dict[str, list[dict[str, Any]]] = {regime: [] for regime in REGIMES}
    seed0_predictions: dict[str, np.ndarray] = {}
    cross_site: dict[str, dict[str, Any]] = {}
    for seed in cfg.seeds:
        models = _fit_all(train, cfg, seed)
        for regime in REGIMES:
            p = _predict_regime(models, regime, cfg, holdout)
            per_seed[regime].append(metrics_by_site(y, p, sites))
            if seed == cfg.seeds[0]:
                seed0_predictions[regime] = p
        if seed == cfg.seeds[0]:
            for source, (theta, scaler) in models["local"].items():
                cross_site[source] = {}
                for target in sorted(set(sites)):
                    rows = [r for r in holdout if r["hospital_id"] == target]
                    cross_site[source][target] = binary_metrics([label(r) for r in rows],
                                                                predict(theta, scaler, cfg.spec, rows))["roc_auc"]
    return {
        "n": len(holdout),
        "positives": int(y.sum()),
        "summary_over_seeds": {regime: _summarize_metric_list(per_seed[regime]) for regime in REGIMES},
        "seed0_detail": {regime: metrics_by_site(y, seed0_predictions[regime], sites) for regime in REGIMES},
        "seed0_auc_bootstrap_ci95": {regime: bootstrap_auc_ci(y, seed0_predictions[regime]) for regime in REGIMES},
        "local_model_cross_site_auc_seed0": cross_site,
    }


def seed_stability(train: list[Record], cfg: ExperimentConfig, seeds: tuple[int, ...],
                   holdout: list[Record] | None = None) -> dict[str, Any]:
    by_site = partition_by_site(train)
    theta_star, _, f_star = centralized_optimum(train, cfg)
    thetas, objectives, aucs, predictions = [], [], [], []
    for seed in seeds:
        fed = train_federated(by_site, cfg, seed, track_objective=True)
        thetas.append(fed["theta"])
        objectives.append(fed["history"][-1].objective)
        if holdout:
            p = predict(fed["theta"], fed["scaler"], cfg.spec, holdout)
            predictions.append(p)
            aucs.append(binary_metrics([label(r) for r in holdout], p)["roc_auc"])
    stacked = np.vstack(thetas)
    result = {
        "seeds": list(seeds),
        "final_objective": summarize(objectives),
        "optimal_objective": f_star,
        "distance_to_optimum": summarize([float(np.linalg.norm(t - theta_star)) for t in thetas]),
        "max_parameter_std": float(stacked.std(axis=0, ddof=1).max()),
    }
    if holdout:
        preds = np.vstack(predictions)
        result["holdout_roc_auc"] = summarize(aucs)
        result["max_prediction_std_across_seeds"] = float(preds.std(axis=0, ddof=1).max())
    return result


def convergence_study(train: list[Record], cfg: ExperimentConfig, epochs_grid: tuple[int, ...] = (1, 2, 5, 10),
                      rounds: int = 60) -> dict[str, Any]:
    by_site = partition_by_site(train)
    theta_star, _, f_star = centralized_optimum(train, cfg)
    results = {}
    for weighting in ("n_samples", "uniform"):
        for epochs in epochs_grid:
            run_cfg = replace(cfg, local_epochs=epochs, rounds=rounds, weighting=weighting)
            fed = train_federated(by_site, run_cfg, seed=0, track_objective=True)
            curve = [h.objective for h in fed["history"]]
            gap = [c - f_star for c in curve]
            within = next((h.round for h, g in zip(fed["history"], gap) if g < 1e-3), None)
            results[f"{weighting}_E{epochs}"] = {
                "weighting": weighting,
                "local_epochs": epochs,
                "objective_curve": [round(c, 6) for c in curve],
                "final_gap_to_pooled_optimum": gap[-1],
                "first_round_within_1e-3": within,
                "distance_to_pooled_optimum": float(np.linalg.norm(fed["theta"] - theta_star)),
            }
    return {"pooled_optimum_objective": f_star, "runs": results}


def select_l2(train: list[Record], cfg: ExperimentConfig, grid: tuple[float, ...]) -> dict[str, Any]:
    """Choose L2 by *federated* site-stratified CV log-loss (no pooling needed)."""
    scores = {}
    for l2 in grid:
        run_cfg = replace(cfg, l2=l2, cv_repeats=2)
        y = np.array([label(r) for r in train])
        losses, aucs = [], []
        for repeat in range(run_cfg.cv_repeats):
            folds = np.array(_stratified_site_folds(train, run_cfg.cv_folds, seed=2000 + repeat))
            oof = np.zeros(len(train))
            for fold in range(run_cfg.cv_folds):
                tr = [r for r, f in zip(train, folds) if f != fold]
                te_idx = np.where(folds == fold)[0]
                fed = train_federated(partition_by_site(tr), run_cfg, seed=repeat)
                oof[te_idx] = predict(fed["theta"], fed["scaler"], run_cfg.spec, [train[i] for i in te_idx])
            m = binary_metrics(y, oof)
            losses.append(m["log_loss"])
            aucs.append(m["roc_auc"])
        scores[str(l2)] = {"cv_log_loss": float(np.mean(losses)), "cv_roc_auc": float(np.mean(aucs))}
    best = min(scores, key=lambda k: scores[k]["cv_log_loss"])
    return {"criterion": "federated site-stratified CV log-loss", "grid": scores, "selected_l2": float(best)}


def non_iid_profile(train: list[Record], cfg: ExperimentConfig) -> dict[str, Any]:
    spec = FeatureSpec.named("extended")
    profile = {}
    for site, rows in partition_by_site(train).items():
        x = spec.matrix(rows)
        means = np.nanmean(x, axis=0)
        profile[site] = {
            "n": len(rows),
            "readmission_prevalence": float(np.mean([label(r) for r in rows])),
            **{name: round(float(v), 3) for name, v in zip(spec.names, means)
               if not name.startswith(("site_", "missing_"))},
        }
    return profile


def dp_study(train: list[Record], cfg: ExperimentConfig, epsilons: tuple[float, ...], delta: float,
             holdout: list[Record] | None = None, rounds: int = 30, local_epochs: int = 1,
             learning_rate: float = 0.5, clip_norm: float = 1.0) -> dict[str, Any]:
    """Utility of record-level DP-FedAvg (full-batch DP-GD locally) vs epsilon."""
    steps = rounds * local_epochs
    dp_cfg = replace(cfg, rounds=rounds, local_epochs=local_epochs, learning_rate=learning_rate)
    gd = replace(dp_cfg.sgd, batch_size=10**9)  # full batch for the non-private control
    rows: dict[str, Any] = {}
    y = np.array([label(r) for r in train])
    for eps in epsilons:
        sigma = None if eps == float("inf") else noise_for_epsilon(eps, steps, delta)
        cv_aucs, cv_briers, holdout_aucs, holdout_briers = [], [], [], []
        for repeat in range(cfg.cv_repeats):
            folds = np.array(_stratified_site_folds(train, cfg.cv_folds, seed=3000 + repeat))
            oof = np.zeros(len(train))
            for fold in range(cfg.cv_folds):
                tr = [r for r, f in zip(train, folds) if f != fold]
                te_idx = np.where(folds == fold)[0]
                dp = DPConfig(noise_multiplier=sigma, clip_norm=clip_norm) if sigma else None
                fed = train_federated(partition_by_site(tr), dp_cfg, seed=repeat, dp=dp, sgd=gd)
                oof[te_idx] = predict(fed["theta"], fed["scaler"], cfg.spec, [train[i] for i in te_idx])
            m = binary_metrics(y, oof)
            cv_aucs.append(m["roc_auc"])
            cv_briers.append(m["brier"])
        if holdout:
            yh = [label(r) for r in holdout]
            for seed in cfg.seeds:
                dp = DPConfig(noise_multiplier=sigma, clip_norm=clip_norm) if sigma else None
                fed = train_federated(partition_by_site(train), dp_cfg, seed=seed, dp=dp, sgd=gd)
                m = binary_metrics(yh, predict(fed["theta"], fed["scaler"], cfg.spec, holdout))
                holdout_aucs.append(m["roc_auc"])
                holdout_briers.append(m["brier"])
        rows["non_private" if sigma is None else f"epsilon_{eps:g}"] = {
            "target_epsilon": None if sigma is None else eps,
            "noise_multiplier": sigma,
            "epsilon_exact_gdp": None if sigma is None else epsilon_exact(sigma, steps, delta),
            "epsilon_rdp_bound": None if sigma is None else epsilon_rdp(sigma, steps, delta),
            "cv_roc_auc": summarize(cv_aucs),
            "cv_brier": summarize(cv_briers),
            "holdout_roc_auc": summarize(holdout_aucs) if holdout else None,
            "holdout_brier": summarize(holdout_briers) if holdout else None,
        }
    return {"delta": delta, "steps_per_record": steps, "rounds": rounds, "local_epochs": local_epochs,
            "learning_rate": learning_rate, "clip_norm": clip_norm, "optimizer": "full-batch local GD",
            "results": rows}


def secure_aggregation_benchmark(train: list[Record], cfg: ExperimentConfig, seed: int = 0) -> dict[str, Any]:
    by_site = partition_by_site(train)
    plain = train_federated(by_site, cfg, seed, secure=False)
    secure = train_federated(by_site, cfg, seed, secure=True)
    dropout_round = max(2, cfg.rounds // 2)
    dropped = train_federated(by_site, cfg, seed, secure=True, track_objective=True,
                              dropout_schedule={dropout_round: frozenset({site_index("HYDERABAD_NODE")})})
    return {
        "max_abs_parameter_difference_secure_vs_plain": float(np.max(np.abs(plain["theta"] - secure["theta"]))),
        "runtime_seconds": {"plain": plain["seconds"], "secure": secure["seconds"]},
        "client_to_server_bytes": {"plain": plain["log"].bytes_by("client->server"),
                                   "secure": secure["log"].bytes_by("client->server")},
        "dropout_test": {
            "dropped_client": "HYDERABAD_NODE",
            "dropout_round": dropout_round,
            "completed_rounds": len(dropped["history"]),
            "participants_after_dropout": dropped["history"][-1].participants,
            "final_objective_on_surviving_nodes": dropped["history"][-1].objective,
        },
    }


def config_dict(cfg: ExperimentConfig) -> dict[str, Any]:
    out = asdict(cfg)
    out["seeds"] = list(cfg.seeds)
    out["features"] = list(cfg.spec.names)
    out["optimizer"] = f"mini-batch SGD (batch {cfg.batch_size}, lr {cfg.learning_rate}), zero-mean N(0, 0.01^2) init"
    return out

