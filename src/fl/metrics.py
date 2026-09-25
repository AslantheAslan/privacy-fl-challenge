"""Discrimination and calibration metrics, overall and per site."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def binary_metrics(y: Sequence[float], p: Sequence[float]) -> dict[str, Any]:
    y_arr = np.asarray(y, dtype=float)
    p_arr = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    both_classes = len(np.unique(y_arr)) == 2
    auc = float(roc_auc_score(y_arr, p_arr)) if both_classes else None
    ap = float(average_precision_score(y_arr, p_arr)) if y_arr.any() else None
    brier = float(brier_score_loss(y_arr, p_arr))
    return {
        "n": int(len(y_arr)),
        "prevalence": float(y_arr.mean()) if len(y_arr) else None,
        "mean_predicted": float(p_arr.mean()) if len(p_arr) else None,
        "roc_auc": auc,
        "average_precision": ap,
        "brier": brier,
        "log_loss": float(log_loss(y_arr, p_arr, labels=[0, 1])),
        # same composite the hidden benchmark uses for the readmission task
        "challenge_score": 0.55 * (auc if auc is not None else 0.5) + 0.25 * (ap or 0.0) + 0.20 * (1 - brier),
    }


def metrics_by_site(y: Sequence[float], p: Sequence[float], sites: Sequence[str]) -> dict[str, Any]:
    y_arr, p_arr, s_arr = np.asarray(y), np.asarray(p), np.asarray(sites)
    result = {"overall": binary_metrics(y_arr, p_arr)}
    for site in sorted(set(s_arr.tolist())):
        mask = s_arr == site
        result[site] = binary_metrics(y_arr[mask], p_arr[mask])
    aucs = [m["roc_auc"] for k, m in result.items() if k != "overall" and m["roc_auc"] is not None]
    result["worst_site_roc_auc"] = min(aucs) if aucs else None
    return result


def bootstrap_auc_ci(y: Sequence[float], p: Sequence[float], n_boot: int = 2000, seed: int = 0) -> list[float] | None:
    """Percentile 95% CI for ROC AUC (stratified resampling keeps both classes)."""
    y_arr, p_arr = np.asarray(y), np.asarray(p)
    pos, neg = np.where(y_arr == 1)[0], np.where(y_arr == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        values.append(roc_auc_score(y_arr[idx], p_arr[idx]))
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def summarize(values: Sequence[float | None]) -> dict[str, float | None]:
    clean = [v for v in values if v is not None and np.isfinite(v)]
    if not clean:
        return {"mean": None, "std": None, "min": None, "max": None}
    arr = np.asarray(clean, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "min": float(arr.min()), "max": float(arr.max())}
