"""Federated learning mechanics: isolation, aggregation rule, equivalences."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.features import FeatureSpec, Standardizer
from src.fl import experiments as ex
from src.fl.client import ClientConfig, HospitalClient
from src.fl.logreg import SGDConfig, fit_newton, local_sgd, objective, predict_proba
from src.fl.server import FederatedServer
from src.fl.transport import InProcessEndpoint, TransportLog, decode_message, encode_message
from src.io_utils import read_jsonl

ROOT = Path(__file__).resolve().parents[1]
TRAIN = read_jsonl(ROOT / "data/train.jsonl")
CFG = ex.ExperimentConfig(rounds=40, seeds=(0,), cv_repeats=1)


def test_transport_roundtrip_and_rejects_arbitrary_objects() -> None:
    payload = {"a": np.arange(6, dtype=np.uint64).reshape(2, 3), "b": b"\x00\xff", "big": 2**300,
               "nested": {1: [1.5, None, "x"]}}
    decoded = decode_message(encode_message(payload))
    assert np.array_equal(decoded["a"], payload["a"]) and decoded["a"].dtype == np.uint64
    assert decoded["b"] == b"\x00\xff" and decoded["big"] == 2**300 and decoded["nested"] == {1: [1.5, None, "x"]}
    with pytest.raises(TypeError):
        encode_message({"callable": print})


def test_client_accepts_only_its_own_site() -> None:
    with pytest.raises(ValueError):
        HospitalClient(ClientConfig(site="BERLIN_NODE"), TRAIN[:10])  # mixes sites


def test_client_release_policy() -> None:
    rows = [r for r in TRAIN if r["hospital_id"] == "CHENNAI_NODE"]
    client = HospitalClient(ClientConfig(site="CHENNAI_NODE"), rows)
    with pytest.raises(ValueError):
        client.handle("compute", {"task": "raw_rows", "round": 1, "participants": [1]})
    with pytest.raises(ValueError):
        client.handle("dump_everything", {})


def test_fedavg_rule_with_stub_clients() -> None:
    """theta = sum n_k theta_k / sum n_k (and uniform: plain mean)."""

    class Stub:
        def __init__(self, cid: int, theta: np.ndarray, n: int):
            self.name, self.cid, self.theta, self.n = f"s{cid}", cid, theta, n

        def request(self, msg_type, payload):
            if msg_type == "hello":
                return {"client_id": self.cid, "secure": False}
            if payload.get("task") == "train":
                w = self.n if payload["weighting"] == "n_samples" else 1.0
                return {"vector": np.concatenate([w * self.theta, [w]])}
            raise AssertionError(msg_type)

        def close(self):
            pass

    stubs = {0: Stub(0, np.array([1.0, 0.0]), 10), 1: Stub(1, np.array([4.0, 3.0]), 30)}
    server = FederatedServer(stubs)
    params = {"theta": np.zeros(2), "epochs": 1, "sgd": {}, "weighting": "n_samples"}
    total = server.aggregate_sum("train", params)
    assert np.allclose(total[:-1] / total[-1], [3.25, 2.25])
    total = server.aggregate_sum("train", {**params, "weighting": "uniform"})
    assert np.allclose(total[:-1] / total[-1], [2.5, 1.5])


def test_single_client_fedavg_equals_local_sgd() -> None:
    rows = [r for r in TRAIN if r["hospital_id"] == "BERLIN_NODE"]
    cfg = replace(CFG, rounds=1, local_epochs=3)
    fed = ex.train_federated({"BERLIN_NODE": rows}, cfg, seed=0)
    spec = cfg.spec
    x_raw = spec.matrix(rows)
    scaler = Standardizer.from_flat_moments(Standardizer.local_moments(x_raw), spec.dim)
    init = np.random.default_rng(0).normal(0.0, 0.01, size=spec.dim + 1)
    y = np.array([r["labels"]["readmission_30d"] for r in rows], float)
    expected = local_sgd(init, scaler.transform(x_raw), y, 3, cfg.sgd, np.random.default_rng([0, 0]))
    assert np.allclose(fed["theta"], expected)


def test_federated_standardisation_equals_pooled_statistics() -> None:
    fed = ex.train_federated(ex.partition_by_site(TRAIN), replace(CFG, rounds=1), seed=0)
    x_raw = CFG.spec.matrix(TRAIN)
    assert np.allclose(fed["scaler"].mean, np.nanmean(x_raw, axis=0))
    assert np.allclose(fed["scaler"].std[fed["scaler"].std != 1.0], np.nanstd(x_raw, axis=0)[fed["scaler"].std != 1.0])


def test_fedavg_converges_to_pooled_optimum_and_secagg_is_lossless() -> None:
    by_site = ex.partition_by_site(TRAIN)
    plain = ex.train_federated(by_site, CFG, seed=0, track_objective=True)
    secure = ex.train_federated(by_site, CFG, seed=0, secure=True, track_objective=True)
    _, _, f_star = ex.centralized_optimum(TRAIN, CFG)
    assert plain["history"][-1].objective - f_star < 2e-3
    assert np.max(np.abs(plain["theta"] - secure["theta"])) < 1e-7


def test_no_row_level_data_leaves_a_client() -> None:
    by_site = ex.partition_by_site(TRAIN)
    for secure in (False, True):
        run = ex.train_federated(by_site, replace(CFG, rounds=3), seed=0, secure=secure, track_objective=True)
        bound = 3 * CFG.spec.dim  # largest aggregate: per-feature (count, sum, sum_sq)
        for entry in run["log"].entries:
            if entry["direction"] != "client->server":
                continue
            _assert_small(entry["fields"], bound)
            if secure and entry["type"] == "compute":
                assert entry["fields"]["vector"]["dtype"] == "<u8"  # masked ring elements only


def _assert_small(fields, bound: int) -> None:
    if isinstance(fields, dict):
        if "array" in fields:
            assert int(np.prod(fields["array"])) <= bound
            return
        for value in fields.values():
            _assert_small(value, bound)
    elif isinstance(fields, list):
        for value in fields:
            _assert_small(value, bound)


def test_secure_training_survives_a_dropout() -> None:
    run = ex.train_federated(ex.partition_by_site(TRAIN), replace(CFG, rounds=6), seed=0, secure=True,
                             track_objective=True, dropout_schedule={3: frozenset({2})})
    assert run["history"][2].participants == [0, 1]
    assert np.isfinite(run["history"][-1].objective)


def test_process_backend_matches_in_process_backend() -> None:
    cfg = replace(CFG, rounds=3)
    by_site = ex.partition_by_site(TRAIN)
    local = ex.train_federated(by_site, cfg, seed=1)
    remote = ex.train_federated(by_site, cfg, seed=1, backend="process", train_path=str(ROOT / "data/train.jsonl"))
    assert np.allclose(local["theta"], remote["theta"])


def test_newton_solution_is_a_stationary_point() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 4))
    y = (rng.random(200) < predict_proba(np.array([1.0, -1.0, 0.5, 0.0, 0.2]), x)).astype(float)
    theta = fit_newton(x, y, l2=0.1)
    eps = 1e-5
    for j in range(len(theta)):
        e = np.zeros_like(theta)
        e[j] = eps
        numeric = (objective(theta + e, x, y, 0.1) - objective(theta - e, x, y, 0.1)) / (2 * eps)
        assert abs(numeric) < 1e-6


def test_site_stratified_folds_keep_every_site_in_every_fold() -> None:
    folds = ex._stratified_site_folds(TRAIN, 5, seed=0)
    for fold in range(5):
        sites = {r["hospital_id"] for r, f in zip(TRAIN, folds) if f == fold}
        assert sites == {"BERLIN_NODE", "CHENNAI_NODE", "HYDERABAD_NODE"}


def test_feature_spec_handles_unknown_site_and_missing_fields() -> None:
    spec = FeatureSpec.named("extended")
    x = spec.matrix([{"hospital_id": "OTHER", "note_text": "", "structured_features": {}}])
    assert x.shape == (1, spec.dim)
    scaler = Standardizer.from_flat_moments(Standardizer.local_moments(spec.matrix(TRAIN)), spec.dim)
    assert np.all(np.isfinite(scaler.transform(x)))
    with pytest.raises(KeyError):
        FeatureSpec.named("nope")


def test_sgd_config_is_serialisable_for_the_wire() -> None:
    from dataclasses import asdict

    assert decode_message(encode_message(asdict(SGDConfig()))) == asdict(SGDConfig())
    assert isinstance(TransportLog().summary(), dict)
    assert InProcessEndpoint  # imported for completeness
