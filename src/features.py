"""Feature engineering for the readmission model.

Features are computed *locally at each hospital node* from its own notes and
structured fields (the NLP extraction runs inside the node), so raw text never
leaves a site. The only cross-site coordination needed is the standardisation
statistics, which are computed federatedly (see ``src.fl.client``).

Two feature sets are defined:

* ``compact`` (default) – selected by repeated 5-fold CV on the training split
  only: demographics/utilisation, site indicator and six comorbidities.
* ``extended`` – adds all nine diagnoses, labs/vitals (with federated mean
  imputation + missingness indicators), medication count and smoking; used as
  an ablation in the experiment report.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from .extraction import extract_clinical_data
from .io_utils import HOSPITALS

DIAGNOSES = (
    "atrial_fibrillation",
    "heart_failure",
    "hypertension",
    "type_2_diabetes",
    "chronic_kidney_disease",
    "coronary_artery_disease",
    "acute_coronary_syndrome",
    "pneumonia",
    "copd",
)
LAB_FIELDS = ("heart_rate_bpm", "systolic_bp_mmhg", "creatinine_mg_dl", "hemoglobin_g_dl", "lvef_percent")

FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "compact": (
        "age_years",
        "prior_admissions_12m",
        "length_of_stay_days",
        "emergency_admission",
        "site_BERLIN_NODE",
        "site_CHENNAI_NODE",
        "site_HYDERABAD_NODE",
        "dx_heart_failure",
        "dx_chronic_kidney_disease",
        "dx_atrial_fibrillation",
        "dx_copd",
        "dx_type_2_diabetes",
        "dx_coronary_artery_disease",
    ),
}
FEATURE_SETS["extended"] = (
    "age_years",
    "sex_male",
    "prior_admissions_12m",
    "length_of_stay_days",
    "emergency_admission",
    *(f"site_{h}" for h in HOSPITALS),
    *(f"dx_{d}" for d in DIAGNOSES),
    *LAB_FIELDS,
    *(f"missing_{f}" for f in LAB_FIELDS),
    "n_medications",
    "smoking_current",
    "smoking_former",
)


@lru_cache(maxsize=8192)
def _cached_extraction(note: str) -> dict[str, Any]:
    return extract_clinical_data(note)


def raw_feature_dict(record: dict[str, Any], extracted: dict[str, Any] | None = None) -> dict[str, float]:
    """All candidate features for one record; labs may be NaN (missing)."""
    extracted = extracted if extracted is not None else _cached_extraction(record.get("note_text", ""))
    structured = record.get("structured_features") or {}
    features: dict[str, float] = {
        "age_years": float(structured.get("age_years") or np.nan),
        "sex_male": float(structured.get("sex") == "male"),
        "prior_admissions_12m": float(structured.get("prior_admissions_12m") or 0),
        "length_of_stay_days": float(structured.get("length_of_stay_days") or 0),
        "emergency_admission": float(bool(structured.get("emergency_admission"))),
        "n_medications": float(len(extracted.get("medications") or [])),
        "smoking_current": float(extracted.get("smoking_status") == "current"),
        "smoking_former": float(extracted.get("smoking_status") == "former"),
    }
    for hospital in HOSPITALS:
        features[f"site_{hospital}"] = float(record.get("hospital_id") == hospital)
    diagnoses = set(extracted.get("diagnoses") or [])
    for diagnosis in DIAGNOSES:
        features[f"dx_{diagnosis}"] = float(diagnosis in diagnoses)
    for field in LAB_FIELDS:
        value = extracted.get(field)
        features[field] = float(value) if value is not None else np.nan
        features[f"missing_{field}"] = float(value is None)
    return features


@dataclass(frozen=True)
class FeatureSpec:
    """Ordered feature list; turns records into a raw (unstandardised) matrix."""

    names: tuple[str, ...]

    @classmethod
    def named(cls, feature_set: str) -> "FeatureSpec":
        if feature_set not in FEATURE_SETS:
            raise KeyError(f"unknown feature set {feature_set!r}; choose from {sorted(FEATURE_SETS)}")
        return cls(FEATURE_SETS[feature_set])

    @property
    def dim(self) -> int:
        return len(self.names)

    def matrix(self, records: list[dict[str, Any]], extracted: list[dict[str, Any]] | None = None) -> np.ndarray:
        rows = []
        for index, record in enumerate(records):
            feats = raw_feature_dict(record, extracted[index] if extracted is not None else None)
            rows.append([feats[name] for name in self.names])
        return np.asarray(rows, dtype=float).reshape(len(records), self.dim)


@dataclass(frozen=True)
class Standardizer:
    """Mean/std scaling with mean imputation of missing values.

    Built from *aggregate* sufficient statistics ``(n, sum, sum_sq, count)`` so
    that the federated and centralised pipelines use identical preprocessing.
    """

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def from_moments(cls, count: np.ndarray, total: np.ndarray, total_sq: np.ndarray) -> "Standardizer":
        count = np.maximum(np.asarray(count, dtype=float), 1.0)
        mean = total / count
        var = np.maximum(total_sq / count - mean**2, 0.0)
        std = np.sqrt(var)
        std[std < 1e-8] = 1.0  # constant columns (e.g. site flag inside one node)
        return cls(mean=mean, std=std)

    @staticmethod
    def local_moments(x_raw: np.ndarray) -> np.ndarray:
        """``[count_j, sum_j, sum_sq_j]`` per column, ignoring NaNs (one flat vector)."""
        observed = ~np.isnan(x_raw)
        filled = np.where(observed, x_raw, 0.0)
        return np.concatenate([observed.sum(axis=0), filled.sum(axis=0), (filled**2).sum(axis=0)]).astype(float)

    @classmethod
    def from_flat_moments(cls, flat: np.ndarray, dim: int) -> "Standardizer":
        return cls.from_moments(flat[:dim], flat[dim : 2 * dim], flat[2 * dim : 3 * dim])

    def transform(self, x_raw: np.ndarray) -> np.ndarray:
        filled = np.where(np.isnan(x_raw), self.mean, x_raw)
        return (filled - self.mean) / self.std
