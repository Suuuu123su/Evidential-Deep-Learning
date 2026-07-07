from __future__ import annotations

import numpy as np


EPS = 1e-12


def alpha_to_prob(alpha: np.ndarray) -> np.ndarray:
    """Expected categorical probability under a Dirichlet distribution."""
    strength = np.sum(alpha, axis=-1, keepdims=True)
    return alpha / np.maximum(strength, EPS)


def alpha_to_mass(alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert Dirichlet parameters to singleton belief masses and ignorance mass.

    For C classes:
        evidence e_c = alpha_c - 1
        S = sum_c alpha_c
        m({c}) = e_c / S
        m(Y) = C / S
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    if alpha.ndim < 1:
        raise ValueError("alpha must have at least one dimension")
    class_count = alpha.shape[-1]
    evidence = np.maximum(alpha - 1.0, 0.0)
    strength = np.sum(alpha, axis=-1)
    belief = evidence / np.maximum(strength[..., None], EPS)
    uncertainty = class_count / np.maximum(strength, EPS)
    return belief, uncertainty


def mass_to_plausibility(belief: np.ndarray, uncertainty: np.ndarray) -> np.ndarray:
    """Compute singleton plausibility pl(c)=m({c})+m(Y)."""
    return belief + uncertainty[..., None]


def alpha_to_plausibility(alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return belief, uncertainty, and plausibility from Dirichlet alpha."""
    belief, uncertainty = alpha_to_mass(alpha)
    plausibility = mass_to_plausibility(belief, uncertainty)
    return belief, uncertainty, plausibility


def predictive_entropy(prob: np.ndarray) -> np.ndarray:
    prob = np.asarray(prob, dtype=np.float64)
    return -np.sum(prob * np.log(np.maximum(prob, EPS)), axis=-1)


def margin(prob: np.ndarray) -> np.ndarray:
    sorted_prob = np.sort(prob, axis=-1)
    return sorted_prob[..., -1] - sorted_prob[..., -2]


def vacuity(alpha: np.ndarray) -> np.ndarray:
    class_count = alpha.shape[-1]
    strength = np.sum(alpha, axis=-1)
    return class_count / np.maximum(strength, EPS)


def edl_expected_nll(prob: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    picked = prob[np.arange(labels.shape[0]), labels]
    return float(np.mean(-np.log(np.maximum(picked, EPS))))

