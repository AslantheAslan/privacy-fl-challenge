"""L2-regularised logistic regression in plain NumPy.

Parameters are a single flat vector ``theta = [w_1..w_d, b]`` so that exactly
the same object is trained locally, averaged by FedAvg, masked by secure
aggregation and fitted centrally. The objective on a dataset of size n is::

    F(theta) = (1/n) * sum_i BCE(y_i, sigmoid(x_i w + b)) + (l2/2) * ||w||^2
               [+ (mu/2) * ||theta - theta_anchor||^2   for FedProx]

The n-weighted average of the per-client objectives equals the pooled
objective, which is what makes the centralised model a fair reference.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z, dtype=float)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def predict_proba(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    return sigmoid(x @ theta[:-1] + theta[-1])


def objective(theta: np.ndarray, x: np.ndarray, y: np.ndarray, l2: float) -> float:
    """Mean binary cross-entropy + L2 on the weights (bias unpenalised)."""
    z = x @ theta[:-1] + theta[-1]
    # log(1 + exp(z)) - y z, computed stably
    bce = np.logaddexp(0.0, z) - y * z
    return float(bce.mean() + 0.5 * l2 * theta[:-1] @ theta[:-1])


def gradient(
    theta: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    l2: float,
    anchor: np.ndarray | None = None,
    mu: float = 0.0,
) -> np.ndarray:
    residual = predict_proba(theta, x) - y
    grad = np.empty_like(theta)
    grad[:-1] = x.T @ residual / len(y) + l2 * theta[:-1]
    grad[-1] = residual.mean()
    if anchor is not None and mu > 0:
        grad += mu * (theta - anchor)
    return grad


def per_example_gradients(theta: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Data-dependent gradient of the BCE term for every example, shape (n, d+1)."""
    residual = (predict_proba(theta, x) - y)[:, None]
    return np.hstack([residual * x, residual])


@dataclass(frozen=True)
class SGDConfig:
    learning_rate: float = 0.2
    batch_size: int = 16
    l2: float = 0.05
    mu: float = 0.0  # FedProx proximal coefficient; 0 = plain FedAvg local solver


def local_sgd(
    theta: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    epochs: int,
    config: SGDConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Mini-batch SGD for ``epochs`` passes; shuffling uses the supplied RNG."""
    theta = theta.copy()
    anchor = theta.copy() if config.mu > 0 else None
    n = len(y)
    if n == 0:
        return theta
    for _ in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, config.batch_size):
            batch = order[start : start + config.batch_size]
            theta -= config.learning_rate * gradient(theta, x[batch], y[batch], config.l2, anchor, config.mu)
    return theta


def fit_newton(x: np.ndarray, y: np.ndarray, l2: float, iterations: int = 50, tol: float = 1e-12) -> np.ndarray:
    """Exact minimiser of ``objective`` (used as the convergence reference)."""
    d = x.shape[1]
    theta = np.zeros(d + 1)
    design = np.hstack([x, np.ones((len(y), 1))])
    penalty = np.full(d + 1, l2)
    penalty[-1] = 0.0
    for _ in range(iterations):
        p = predict_proba(theta, x)
        grad = design.T @ (p - y) / len(y) + penalty * theta
        weights = p * (1 - p) / len(y)
        hessian = design.T @ (design * weights[:, None]) + np.diag(penalty) + 1e-10 * np.eye(d + 1)
        step = np.linalg.solve(hessian, grad)
        theta -= step
        if np.linalg.norm(step) < tol:
            break
    return theta
