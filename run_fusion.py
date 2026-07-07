from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from edl_cifar5.fusion import apply_fusion_rule


def accuracy(prob: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(np.argmax(prob, axis=-1) == labels))


def nll(prob: np.ndarray, labels: np.ndarray) -> float:
    picked = prob[np.arange(labels.shape[0]), labels]
    return float(np.mean(-np.log(np.maximum(picked, 1e-12))))


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
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if not np.any(mask):
            continue
        score += np.sum(mask) / total * abs(float(np.mean(correct[mask]) - np.mean(conf[mask])))
    return float(score)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate five fusion rules on exported EDL outputs.")
    parser.add_argument("--outputs", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--rho", type=float, default=0.7)
    parser.add_argument("--r", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = np.load(args.outputs, allow_pickle=True)
    prob = data["prob"]
    plausibility = data["plausibility"]
    labels = data["labels"].astype(np.int64)
    source_count = prob.shape[1]
    r = args.r if args.r is not None else source_count // 2 + 1

    rules = {
        "conjunctive": {"name": "conjunctive"},
        "disjunctive": {"name": "disjunctive"},
        f"r_out_of_{source_count}_r{r}": {"name": "r_out_of_k", "r": r},
        "discount": {"name": "discount", "rho": args.rho},
        "average": {"name": "average"},
    }

    metrics = {}
    for display_name, config in rules.items():
        fused = apply_fusion_rule(
            config["name"],
            prob,
            plausibility,
            r=config.get("r"),
            rho=config.get("rho", args.rho),
        )
        metrics[display_name] = {
            "accuracy": accuracy(fused, labels),
            "nll": nll(fused, labels),
            "brier": brier(fused, labels),
            "ece": ece(fused, labels),
        }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

