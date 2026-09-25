"""Differential-privacy mechanism and accounting."""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.fl.logreg import SGDConfig, gradient
from src.privacy.dp import (
    DPConfig,
    dp_local_gd,
    epsilon_exact,
    epsilon_rdp,
    gdp_delta,
    noise_for_epsilon,
)


def test_gdp_delta_matches_closed_form_at_zero_epsilon() -> None:
    # delta(0) = Phi(mu/2) - Phi(-mu/2) = 2*Phi(mu/2) - 1
    assert gdp_delta(1.0, 0.0) == pytest.approx(math.erf(0.5 / math.sqrt(2)), rel=1e-9)


@pytest.mark.parametrize("sigma", [1.0, 3.0, 10.0])
def test_exact_accounting_is_never_looser_than_rdp(sigma: float) -> None:
    assert epsilon_exact(sigma, 30, 1e-3) <= epsilon_rdp(sigma, 30, 1e-3) + 1e-9


def test_epsilon_decreases_with_noise_and_increases_with_steps() -> None:
    assert epsilon_exact(2.0, 30, 1e-3) > epsilon_exact(4.0, 30, 1e-3)
    assert epsilon_exact(4.0, 60, 1e-3) > epsilon_exact(4.0, 30, 1e-3)


@pytest.mark.parametrize("target", [0.5, 1.0, 4.0, 8.0])
def test_noise_calibration_inverts_accounting(target: float) -> None:
    sigma = noise_for_epsilon(target, 30, 1e-3)
    assert epsilon_exact(sigma, 30, 1e-3) == pytest.approx(target, rel=1e-3)


def test_dp_gd_without_noise_or_clipping_is_plain_gradient_descent() -> None:
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(40, 3)), (rng.random(40) < 0.4).astype(float)
    theta0 = np.zeros(4)
    sgd = SGDConfig(learning_rate=0.3, l2=0.05)
    dp_theta = dp_local_gd(theta0, x, y, 5, sgd, DPConfig(noise_multiplier=1e-12, clip_norm=1e6), rng)
    theta = theta0.copy()
    for _ in range(5):
        theta -= 0.3 * gradient(theta, x, y, 0.05)
    assert np.allclose(dp_theta, theta, atol=1e-9)


def test_clipping_bounds_each_records_influence() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(20, 3)) * 100.0  # huge gradients before clipping
    y = np.ones(20)
    sgd = SGDConfig(learning_rate=1.0, l2=0.0)
    step = dp_local_gd(np.zeros(4), x, y, 1, sgd, DPConfig(noise_multiplier=1e-12, clip_norm=0.5), rng)
    # mean of 20 clipped gradients (each norm <= 0.5) has norm <= 0.5
    assert np.linalg.norm(step) <= 0.5 + 1e-9
