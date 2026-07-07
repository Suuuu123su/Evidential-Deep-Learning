from __future__ import annotations

from itertools import combinations

import numpy as np


EPS = 1e-12


def accuracy(prob: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(np.argmax(prob, axis=-1) == labels))


def nll(prob: np.ndarray, labels: np.ndarray) -> float:
    picked = np.maximum(prob[np.arange(labels.shape[0]), labels], EPS)
    return float(np.mean(-np.log(picked)))


def brier(prob: np.ndarray, labels: np.ndarray) -> float:
    y = np.eye(prob.shape[-1])[labels]
    return float(np.mean(np.sum((prob - y) ** 2, axis=-1)))


def ece(prob: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    conf = np.max(prob, axis=-1)
    pred = np.argmax(prob, axis=-1)
    correct = (pred == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = labels.shape[0]
    score = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf >= lo) & (conf <= hi) if i == 0 else (conf > lo) & (conf <= hi)
        if np.any(mask):
            score += np.sum(mask) / total * abs(float(np.mean(correct[mask]) - np.mean(conf[mask])))
    return float(score)


def metric_row(prob: np.ndarray, labels: np.ndarray, *, bins: int = 15) -> dict[str, float]:
    return {
        "accuracy": accuracy(prob, labels),
        "nll": nll(prob, labels),
        "brier": brier(prob, labels),
        "ece": ece(prob, labels, bins=bins),
    }


def per_class_accuracy(prob: np.ndarray, labels: np.ndarray, class_count: int) -> dict[int, float]:
    pred = np.argmax(prob, axis=-1)
    values: dict[int, float] = {}
    for class_id in range(class_count):
        mask = labels == class_id
        values[class_id] = float(np.mean(pred[mask] == labels[mask])) if np.any(mask) else float("nan")
    return values


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size < 2 or np.std(a) < EPS or np.std(b) < EPS:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def pairwise_diversity(prob: np.ndarray, plausibility: np.ndarray, labels: np.ndarray) -> dict[str, object]:
    if prob.ndim != 3:
        raise ValueError("prob must have shape [N,K,C]")
    pred = np.argmax(prob, axis=-1)
    correct = pred == labels[:, None]
    pair_rows = []
    for i, j in combinations(range(prob.shape[1]), 2):
        wrong_i = ~correct[:, i]
        wrong_j = ~correct[:, j]
        same_prediction = pred[:, i] == pred[:, j]
        pair_rows.append(
            {
                "model_i": i,
                "model_j": j,
                "disagreement": float(np.mean(pred[:, i] != pred[:, j])),
                "double_fault": float(np.mean(wrong_i & wrong_j)),
                "same_wrong_prediction": float(np.mean(wrong_i & wrong_j & same_prediction)),
                "correctness_corr": _corr(correct[:, i].astype(float), correct[:, j].astype(float)),
                "prob_corr": _corr(prob[:, i, :], prob[:, j, :]),
                "plausibility_corr": _corr(plausibility[:, i, :], plausibility[:, j, :]),
            }
        )

    summary: dict[str, object] = {"pairs": pair_rows}
    if pair_rows:
        for key in (
            "disagreement",
            "double_fault",
            "same_wrong_prediction",
            "correctness_corr",
            "prob_corr",
            "plausibility_corr",
        ):
            values = np.asarray([row[key] for row in pair_rows], dtype=np.float64)
            summary[f"mean_{key}"] = float(np.nanmean(values))
    return summary
