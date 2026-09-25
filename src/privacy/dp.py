"""Record-level differential privacy for local training (DP-GD) and accounting.

Mechanism
---------
Each local step is full-batch gradient descent on the node's own rows:

    g = (1 / n_k) * ( sum_i clip_C(grad_i) + N(0, sigma^2 C^2 I) ) + l2 * w

``clip_C`` rescales each per-example gradient to L2 norm <= C. The L2 term is
data-independent and needs no noise. Under add/remove-one-record adjacency
(n_k treated as a public constant, e.g. a published site size) the noisy sum
is a Gaussian mechanism with sensitivity C, hence each step is
``(1/sigma)``-GDP. There is no subsampling, so no amplification argument is
needed and the accounting below is exact rather than an upper bound.

Composition
-----------
A record only influences its own node's T = rounds x local_epochs steps.
T adaptive Gaussian mechanisms compose to mu-GDP with mu = sqrt(T) / sigma
(Dong, Roth & Su 2019), which converts *exactly* to (epsilon, delta)-DP via

    delta(eps) = Phi(-eps/mu + mu/2) - exp(eps) * Phi(-eps/mu - mu/2).

We also report the Rényi-DP bound (Mironov 2017) with the conversion of
Balle et al. (2020) as an independent cross-check (it is always >= the exact
value). Everything downstream (FedAvg averaging, secure aggregation, the final
model and its predictions) is post-processing and inherits the guarantee.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..fl.logreg import SGDConfig, per_example_gradients


@dataclass(frozen=True)
class DPConfig:
    noise_multiplier: float
    clip_norm: float = 1.0


def dp_local_gd(
    theta: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    epochs: int,
    sgd: SGDConfig,
    dp: DPConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    theta = theta.copy()
    n = len(y)
    for _ in range(epochs):
        grads = per_example_gradients(theta, x, y)
        norms = np.linalg.norm(grads, axis=1, keepdims=True)
        clipped = grads / np.maximum(1.0, norms / dp.clip_norm)
        noisy_sum = clipped.sum(axis=0) + rng.normal(0.0, dp.noise_multiplier * dp.clip_norm, size=theta.shape)
        step = noisy_sum / n
        step[:-1] += sgd.l2 * theta[:-1]
        theta -= sgd.learning_rate * step
    return theta


# ---------------------------------------------------------------------------
# Accounting
# ---------------------------------------------------------------------------
def _phi(value: float) -> float:
    return 0.5 * math.erfc(-value / math.sqrt(2.0))


def gdp_mu(noise_multiplier: float, steps: int) -> float:
    return math.sqrt(steps) / noise_multiplier


def gdp_delta(mu: float, epsilon: float) -> float:
    return _phi(-epsilon / mu + mu / 2.0) - math.exp(epsilon) * _phi(-epsilon / mu - mu / 2.0)


def epsilon_exact(noise_multiplier: float, steps: int, delta: float) -> float:
    """Smallest epsilon such that the mechanism is (epsilon, delta)-DP (bisection)."""
    mu = gdp_mu(noise_multiplier, steps)
    low, high = 0.0, 1.0
    while gdp_delta(mu, high) > delta:
        high *= 2.0
        if high > 1e4:
            return math.inf
    for _ in range(200):
        mid = (low + high) / 2.0
        if gdp_delta(mu, mid) > delta:
            low = mid
        else:
            high = mid
    return high


def epsilon_rdp(noise_multiplier: float, steps: int, delta: float) -> float:
    """RDP upper bound: eps(a) = T a / (2 sigma^2); Balle et al. (2020) conversion."""
    best = math.inf
    for order in np.concatenate([np.linspace(1.01, 10, 400), np.linspace(10, 512, 400)]):
        rdp = steps * order / (2.0 * noise_multiplier**2)
        eps = rdp + math.log((order - 1) / order) - (math.log(delta) + math.log(order)) / (order - 1)
        best = min(best, eps)
    return max(best, 0.0)


def noise_for_epsilon(target_epsilon: float, steps: int, delta: float) -> float:
    """Smallest noise multiplier achieving ``target_epsilon`` under exact accounting."""
    low, high = 1e-3, 1.0
    while epsilon_exact(high, steps, delta) > target_epsilon:
        high *= 2.0
    for _ in range(100):
        mid = (low + high) / 2.0
        if epsilon_exact(mid, steps, delta) > target_epsilon:
            low = mid
        else:
            high = mid
    return high
