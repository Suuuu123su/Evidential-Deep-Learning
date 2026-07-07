from __future__ import annotations

import numpy as np


EPS = 1e-12


def reliable(belief: np.ndarray, uncertainty: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """R_i: source is reliable; keep masses unchanged."""
    return belief, uncertainty, belief + uncertainty[..., None]


def discounted(
    belief: np.ndarray,
    uncertainty: np.ndarray,
    rho: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """D_i(rho): source is partially reliable."""
    rho = float(np.clip(rho, 0.0, 1.0))
    belief_d = rho * belief
    uncertainty_d = 1.0 - rho + rho * uncertainty
    return belief_d, uncertainty_d, belief_d + uncertainty_d[..., None]


def vacuous_like(belief: np.ndarray, uncertainty: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """V_i: source is invalid/abstained; all mass goes to Y."""
    belief_v = np.zeros_like(belief)
    uncertainty_v = np.ones_like(uncertainty)
    plausibility_v = np.ones_like(belief)
    return belief_v, uncertainty_v, plausibility_v


def approximately_reliable(
    belief: np.ndarray,
    uncertainty: np.ndarray,
    adjacency: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """AR_i: expand singleton mass to neighboring/confusable classes.

    adjacency[a,b]=1 means mass assigned to class a also supports class b.
    """
    adjacency = np.asarray(adjacency, dtype=np.float64)
    class_count = belief.shape[-1]
    if adjacency.shape != (class_count, class_count):
        raise ValueError(f"adjacency must have shape [{class_count},{class_count}]")
    adjacency = np.clip(adjacency, 0.0, 1.0)
    belief_ar = belief @ adjacency
    belief_ar = np.minimum(belief_ar, 1.0)
    return belief_ar, uncertainty, np.minimum(belief_ar + uncertainty[..., None], 1.0)


def bias_corrected(
    belief: np.ndarray,
    uncertainty: np.ndarray,
    confusion: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """B_i(Q_i): correct systematic bias with a row-stochastic confusion matrix.

    confusion[a,b] = P(true class b | source predicts/supports a)
    """
    confusion = np.asarray(confusion, dtype=np.float64)
    class_count = belief.shape[-1]
    if confusion.shape != (class_count, class_count):
        raise ValueError(f"confusion must have shape [{class_count},{class_count}]")
    row_sums = np.maximum(confusion.sum(axis=1, keepdims=True), EPS)
    confusion = confusion / row_sums
    belief_b = belief @ confusion
    return belief_b, uncertainty, np.minimum(belief_b + uncertainty[..., None], 1.0)


SOURCE_BEHAVIORS = {
    "R": reliable,
    "D": discounted,
    "V": vacuous_like,
    "AR": approximately_reliable,
    "B": bias_corrected,
}


def default_adjacency(class_count: int) -> np.ndarray:
    """Identity adjacency; replace with semantic class-neighbor knowledge if available."""
    return np.eye(class_count, dtype=np.float64)


def default_confusion(class_count: int) -> np.ndarray:
    """Identity bias matrix; replace with validation-set confusion estimates if available."""
    return np.eye(class_count, dtype=np.float64)

