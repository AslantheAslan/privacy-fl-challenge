"""FedAvg server / orchestrator.

The server holds no patient data. It only (i) broadcasts the global model and
configuration, (ii) receives per-round *aggregates* from each node and (iii)
combines them. With ``secure=True`` it additionally drives the secure
aggregation protocol and only ever learns the *sum* of the nodes' vectors.

Aggregation rule (McMahan et al., 2017)::

    theta_{t+1} = sum_k w_k * theta_k^{t+1} / sum_k w_k,   w_k = n_k  (default)
                                                          w_k = 1    ("uniform")

Each node sends ``[w_k * theta_k, w_k]`` so that the weighted average can be
computed from a single secure sum; neither the individual models nor even the
individual site sizes are revealed to the server under secure aggregation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from ..features import Standardizer
from ..privacy.dp import DPConfig
from ..privacy.secagg import SecAggServer
from .logreg import SGDConfig
from .transport import Endpoint


@dataclass
class RoundRecord:
    round: int
    objective: float | None
    update_norm: float
    participants: list[int]


@dataclass
class FedAvgResult:
    theta: np.ndarray
    scaler: Standardizer
    history: list[RoundRecord] = field(default_factory=list)


class FederatedServer:
    def __init__(self, endpoints: dict[int, Endpoint], secure: bool = False, threshold: int = 2):
        self.endpoints = endpoints
        self.secure = secure
        self.threshold = threshold
        self.active = sorted(endpoints)
        self._round = 0
        self._secagg: SecAggServer | None = None
        for client_id, endpoint in endpoints.items():
            info = endpoint.request("hello", {})
            if info["client_id"] != client_id or info["secure"] != secure:
                raise RuntimeError(f"client {client_id} configuration mismatch: {info}")

    # ------------------------------------------------------------------
    def setup_secure_aggregation(self) -> None:
        """S0/S1: exchange public keys and route encrypted mask-key shares."""
        if not self.secure:
            return
        keys = {cid: self.endpoints[cid].request("secagg_advertise", {}) for cid in self.active}
        for cid in self.active:
            self.endpoints[cid].request("secagg_keys", {"keys": keys})
        self._secagg = SecAggServer(self.threshold, keys)
        self._route_shares("mask_key", 0, "secagg_share_mask_key", {})

    def _route_shares(self, kind: str, round_id: int, request: str, payload: dict[str, Any]) -> None:
        outboxes = {cid: self.endpoints[cid].request(request, payload)["ciphertexts"] for cid in self.active}
        for recipient in self.active:
            inbound = {sender: outboxes[sender][recipient] for sender in self.active if sender != recipient}
            self.endpoints[recipient].request("secagg_deliver", {"kind": kind, "round": round_id, "ciphertexts": inbound})

    def aggregate_sum(self, task: str, params: dict[str, Any], dropouts: frozenset[int] = frozenset()) -> np.ndarray:
        """Sum of the nodes' vectors for ``task``; plaintext or secure."""
        self._round += 1
        round_id = self._round
        participants = list(self.active)
        request = {"task": task, "round": round_id, "participants": participants, **params}
        if not self.secure:
            vectors = [self.endpoints[c].request("compute", request)["vector"] for c in participants if c not in dropouts]
            return np.sum(vectors, axis=0)
        if self._secagg is None:
            raise RuntimeError("call setup_secure_aggregation() first")
        self._route_shares("self_mask", round_id, "secagg_share_self_mask", {"round": round_id})
        masked = {c: self.endpoints[c].request("compute", request)["vector"] for c in participants if c not in dropouts}
        survivors = sorted(masked)
        dropped = sorted(set(participants) - set(survivors))
        responses = {
            c: self.endpoints[c].request("secagg_unmask", {"round": round_id, "survivors": survivors, "dropped": dropped})
            for c in survivors
        }
        total = self._secagg.unmask(round_id, masked, participants, responses)
        # A dropped node's pairwise-mask key has been reconstructed; it must
        # re-key before it can participate again, so it leaves the cohort.
        self.active = survivors
        return total

    # ------------------------------------------------------------------
    def federated_standardizer(self, dim: int) -> Standardizer:
        moments = self.aggregate_sum("moments", {})
        scaler = Standardizer.from_flat_moments(moments, dim)
        for cid in self.active:
            self.endpoints[cid].request("set_scaler", {"mean": scaler.mean, "std": scaler.std})
        return scaler

    def global_objective(self, theta: np.ndarray, l2: float) -> float:
        total = self.aggregate_sum("loss", {"theta": theta, "l2": l2})
        return float(total[0] / total[1])

    def fit(
        self,
        dim: int,
        rounds: int,
        local_epochs: int,
        sgd: SGDConfig,
        seed: int,
        weighting: str = "n_samples",
        dp: DPConfig | None = None,
        track_objective: bool = True,
        dropout_schedule: dict[int, frozenset[int]] | None = None,
    ) -> FedAvgResult:
        if weighting not in ("n_samples", "uniform"):
            raise ValueError("weighting must be 'n_samples' or 'uniform'")
        self.setup_secure_aggregation()
        scaler = self.federated_standardizer(dim)
        theta = np.random.default_rng(seed).normal(0.0, 0.01, size=dim + 1)
        history: list[RoundRecord] = []
        for t in range(1, rounds + 1):
            params: dict[str, Any] = {"theta": theta, "epochs": local_epochs, "sgd": asdict(sgd), "weighting": weighting}
            if dp is not None:
                params["dp"] = asdict(dp)
            total = self.aggregate_sum("train", params, (dropout_schedule or {}).get(t, frozenset()))
            new_theta = total[:-1] / total[-1]
            update_norm = float(np.linalg.norm(new_theta - theta))
            theta = new_theta
            value = self.global_objective(theta, sgd.l2) if track_objective else None
            history.append(RoundRecord(t, value, update_norm, list(self.active)))
        return FedAvgResult(theta=theta, scaler=scaler, history=history)

    def close(self) -> None:
        for endpoint in self.endpoints.values():
            endpoint.close()
