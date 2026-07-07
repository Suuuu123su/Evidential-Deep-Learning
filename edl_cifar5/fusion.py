from __future__ import annotations

import numpy as np


EPS = 1e-12


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    scores = np.maximum(scores, EPS)
    return scores / np.sum(scores, axis=-1, keepdims=True)


def conjunctive(plausibility: np.ndarray) -> np.ndarray:
    """Conjunctive score: all sources must support the class."""
    return normalize_scores(np.prod(plausibility, axis=1))


def disjunctive(plausibility: np.ndarray) -> np.ndarray:
    """Disjunctive score: at least one source supports the class."""
    return normalize_scores(1.0 - np.prod(1.0 - plausibility, axis=1))


def r_out_of_k(plausibility: np.ndarray, r: int) -> np.ndarray:
    """Poisson-binomial r-out-of-K fusion over class-wise source supports.

    plausibility shape: [N,K,C]
    output shape: [N,C]
    """
    plausibility = np.asarray(plausibility, dtype=np.float64)
    if plausibility.ndim != 3:
        raise ValueError("plausibility must have shape [N,K,C]")
    sample_count, source_count, class_count = plausibility.shape
    if not 1 <= r <= source_count:
        raise ValueError(f"r must be in [1,{source_count}], got {r}")

    scores = np.zeros((sample_count, class_count), dtype=np.float64)
    for n in range(sample_count):
        for c in range(class_count):
            dp = np.zeros(source_count + 1, dtype=np.float64)
            dp[0] = 1.0
            for a in plausibility[n, :, c]:
                next_dp = np.zeros_like(dp)
                for j in range(source_count):
                    next_dp[j] += dp[j] * (1.0 - a)
                    next_dp[j + 1] += dp[j] * a
                dp = next_dp
            scores[n, c] = np.sum(dp[r:])
    return normalize_scores(scores)


def discount_plausibility(plausibility: np.ndarray, rho: np.ndarray | float) -> np.ndarray:
    """Apply Shafer-style reliability discount on singleton plausibilities.

    pl_i'(c)=1-rho_i+rho_i*pl_i(c)
    """
    plausibility = np.asarray(plausibility, dtype=np.float64)
    rho_array = np.asarray(rho, dtype=np.float64)
    if rho_array.ndim == 0:
        rho_array = np.full(plausibility.shape[1], float(rho_array))
    if rho_array.shape != (plausibility.shape[1],):
        raise ValueError(f"rho must be scalar or shape [{plausibility.shape[1]}]")
    rho_array = np.clip(rho_array, 0.0, 1.0)
    return 1.0 - rho_array[None, :, None] + rho_array[None, :, None] * plausibility


def discount_conjunctive(plausibility: np.ndarray, rho: np.ndarray | float) -> np.ndarray:
    return conjunctive(discount_plausibility(plausibility, rho))


def average(prob: np.ndarray) -> np.ndarray:
    """Average expected class probabilities over sources."""
    prob = np.asarray(prob, dtype=np.float64)
    if prob.ndim != 3:
        raise ValueError("prob must have shape [N,K,C]")
    return normalize_scores(np.mean(prob, axis=1))


def apply_fusion_rule(
    name: str,
    prob: np.ndarray,
    plausibility: np.ndarray,
    *,
    r: int | None = None,
    rho: np.ndarray | float = 0.7,
) -> np.ndarray:
    if name == "conjunctive":
        return conjunctive(plausibility)
    if name == "disjunctive":
        return disjunctive(plausibility)
    if name == "r_out_of_k":
        if r is None:
            r = plausibility.shape[1] // 2 + 1
        return r_out_of_k(plausibility, r=r)
    if name == "discount":
        return discount_conjunctive(plausibility, rho=rho)
    if name == "average":
        return average(prob)
    raise ValueError(f"Unknown fusion rule: {name}")

