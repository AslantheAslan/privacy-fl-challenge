"""Hospital node: owns its patient rows and decides what leaves the site.

A ``HospitalClient`` is constructed from its *own* records only. It runs the
NLP extraction and feature computation locally, keeps the design matrix and
labels in private attributes, and answers a small set of server requests.
Every numeric reply is an **aggregate** (moments, n-weighted parameters, loss
sums) — never a row. When secure aggregation is enabled, the client refuses to
release any aggregate in the clear: replies are masked vectors in Z_{2^64}.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..features import FeatureSpec, Standardizer
from ..io_utils import HOSPITALS, read_jsonl
from ..privacy.dp import DPConfig, dp_local_gd
from ..privacy.secagg import SecAggClient
from .logreg import SGDConfig, local_sgd, objective


def site_index(site: str) -> int:
    return HOSPITALS.index(site) if site in HOSPITALS else len(HOSPITALS)


@dataclass(frozen=True)
class ClientConfig:
    """Picklable description used to start a client in its own process."""

    site: str
    feature_set: str = "compact"
    seed: int = 0
    secure: bool = False
    threshold: int = 2
    train_path: str | None = None
    row_ids: tuple[str, ...] | None = None  # optional subset (cross-validation folds)


class HospitalClient:
    ALLOWED_TASKS = ("moments", "train", "loss")

    def __init__(self, config: ClientConfig, records: list[dict[str, Any]]):
        if any(r.get("hospital_id") != config.site for r in records):
            raise ValueError("a hospital client may only hold its own site's records")
        self.config = config
        self.site = config.site
        self.client_id = site_index(config.site)
        self._spec = FeatureSpec.named(config.feature_set)
        # NLP extraction runs here, inside the node: notes never leave the site.
        self._x_raw = self._spec.matrix(records)
        self._y = np.array([float(r["labels"]["readmission_30d"]) for r in records])
        self._x: np.ndarray | None = None
        self._rng = np.random.default_rng([config.seed, self.client_id])
        self._secagg = SecAggClient(self.client_id, config.threshold) if config.secure else None

    @classmethod
    def from_file(cls, config: ClientConfig) -> HospitalClient:
        """Load only this site's rows from the shared training file."""
        if config.train_path is None:
            raise ValueError("train_path required")
        records = [r for r in read_jsonl(Path(config.train_path)) if r.get("hospital_id") == config.site]
        if config.row_ids is not None:
            wanted = set(config.row_ids)
            records = [r for r in records if r["case_id"] in wanted]
        return cls(config, records)

    # ------------------------------------------------------------------
    @property
    def n_samples(self) -> int:
        return len(self._y)

    def handle(self, msg_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        handlers = {
            "hello": self._hello,
            "secagg_advertise": lambda p: self._require_secagg().advertise_keys(),
            "secagg_keys": self._secagg_keys,
            "secagg_share_mask_key": lambda p: {"ciphertexts": self._require_secagg().share_mask_key()},
            "secagg_deliver": self._secagg_deliver,
            "secagg_share_self_mask": lambda p: {"ciphertexts": self._require_secagg().share_self_mask(p["round"])},
            "secagg_unmask": lambda p: self._require_secagg().unmask_response(p["round"], p["survivors"], p["dropped"]),
            "set_scaler": self._set_scaler,
            "compute": self._compute,
        }
        if msg_type not in handlers:
            raise ValueError(f"unsupported request {msg_type!r}")
        return handlers[msg_type](payload) or {}

    # -- protocol helpers -------------------------------------------------
    def _hello(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"client_id": self.client_id, "site": self.site, "secure": self._secagg is not None}

    def _require_secagg(self) -> SecAggClient:
        if self._secagg is None:
            raise RuntimeError("secure aggregation is not enabled on this client")
        return self._secagg

    def _secagg_keys(self, payload: dict[str, Any]) -> None:
        self._require_secagg().receive_public_keys({int(k): v for k, v in payload["keys"].items()})

    def _secagg_deliver(self, payload: dict[str, Any]) -> None:
        self._require_secagg().receive_shares(payload["kind"], payload["round"],
                                              {int(k): v for k, v in payload["ciphertexts"].items()})

    def _set_scaler(self, payload: dict[str, Any]) -> None:
        self._x = Standardizer(np.asarray(payload["mean"]), np.asarray(payload["std"])).transform(self._x_raw)

    # -- aggregate computations --------------------------------------------
    def _compute(self, payload: dict[str, Any]) -> dict[str, Any]:
        task = payload["task"]
        if task not in self.ALLOWED_TASKS:
            raise ValueError(f"task {task!r} is not permitted by this node's release policy")
        vector = getattr(self, f"_task_{task}")(payload)
        if self._secagg is not None:
            return {"vector": self._secagg.masked_input(payload["round"], vector, payload["participants"])}
        return {"vector": vector}

    def _task_moments(self, payload: dict[str, Any]) -> np.ndarray:
        return Standardizer.local_moments(self._x_raw)

    def _task_train(self, payload: dict[str, Any]) -> np.ndarray:
        if self._x is None:
            raise RuntimeError("scaler not set")
        theta = np.asarray(payload["theta"], dtype=float)
        sgd = SGDConfig(**payload["sgd"])
        if payload.get("dp"):
            theta_k = dp_local_gd(theta, self._x, self._y, payload["epochs"], sgd, DPConfig(**payload["dp"]), self._rng)
        else:
            theta_k = local_sgd(theta, self._x, self._y, payload["epochs"], sgd, self._rng)
        weight = float(self.n_samples) if payload.get("weighting", "n_samples") == "n_samples" else 1.0
        return np.concatenate([weight * theta_k, [weight]])

    def _task_loss(self, payload: dict[str, Any]) -> np.ndarray:
        if self._x is None:
            raise RuntimeError("scaler not set")
        theta = np.asarray(payload["theta"], dtype=float)
        n = float(self.n_samples)
        return np.array([n * objective(theta, self._x, self._y, float(payload["l2"])), n])

    # -- purely local training (the "local model" baseline) ------------------
    def fit_local(self, epochs: int, sgd: SGDConfig, init: np.ndarray) -> tuple[np.ndarray, Standardizer]:
        """Train on this node alone, with *local* standardisation statistics."""
        flat = Standardizer.local_moments(self._x_raw)
        scaler = Standardizer.from_flat_moments(flat, self._spec.dim)
        x = scaler.transform(self._x_raw)
        theta = local_sgd(init, x, self._y, epochs, sgd, self._rng)
        return theta, scaler


def build_client(config: ClientConfig) -> HospitalClient:
    """Top-level factory (picklable) used by the process endpoint."""
    return HospitalClient.from_file(config)


def sgd_payload(config: SGDConfig) -> dict[str, float]:
    return asdict(config)
