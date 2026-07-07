from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate r-out-of-K split manifests.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--expected-model-train-per-class", type=int, default=4000)
    parser.add_argument("--expected-checkpoint-val-per-class", type=int, default=500)
    parser.add_argument("--expected-fusion-val-per-class", type=int, default=500)
    parser.add_argument("--expected-per-model-train-size", type=int, default=4000)
    parser.add_argument("--overlap-tolerance", type=float, default=1e-9)
    return parser.parse_args()


def load_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "splits.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing split manifest: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    manifest = load_manifest(run_dir)

    expected_model_train_size = args.expected_model_train_per_class * len(manifest["class_stats"])
    expected_checkpoint_val_size = args.expected_checkpoint_val_per_class * len(manifest["class_stats"])
    expected_fusion_val_size = args.expected_fusion_val_per_class * len(manifest["class_stats"])

    for row in manifest["class_stats"]:
        class_id = row["class_id"]
        require(
            int(row["model_train_count"]) == args.expected_model_train_per_class,
            f"class {class_id} model_train_count={row['model_train_count']}",
        )
        require(
            int(row["checkpoint_val_count"]) == args.expected_checkpoint_val_per_class,
            f"class {class_id} checkpoint_val_count={row['checkpoint_val_count']}",
        )
        require(
            int(row["fusion_val_count"]) == args.expected_fusion_val_per_class,
            f"class {class_id} fusion_val_count={row['fusion_val_count']}",
        )

    model_train_values = [int(x) for x in manifest["model_train_indices"]]
    checkpoint_val_values = [int(x) for x in manifest["checkpoint_val_indices"]]
    fusion_val_values = [int(x) for x in manifest["fusion_val_indices"]]
    require(len(model_train_values) == expected_model_train_size, f"model_train_size={len(model_train_values)}")
    require(
        len(checkpoint_val_values) == expected_checkpoint_val_size,
        f"checkpoint_val_size={len(checkpoint_val_values)}",
    )
    require(len(fusion_val_values) == expected_fusion_val_size, f"fusion_val_size={len(fusion_val_values)}")
    require(len(set(model_train_values)) == len(model_train_values), "model_train contains duplicate indices")
    require(
        len(set(checkpoint_val_values)) == len(checkpoint_val_values),
        "checkpoint_val contains duplicate indices",
    )
    require(len(set(fusion_val_values)) == len(fusion_val_values), "fusion_val contains duplicate indices")

    model_train = set(model_train_values)
    checkpoint_val = set(checkpoint_val_values)
    fusion_val = set(fusion_val_values)
    require(model_train.isdisjoint(checkpoint_val), "model_train overlaps checkpoint_val")
    require(model_train.isdisjoint(fusion_val), "model_train overlaps fusion_val")
    require(checkpoint_val.isdisjoint(fusion_val), "checkpoint_val overlaps fusion_val")

    model_indices = [[int(x) for x in values] for values in manifest["model_indices"]]
    for model_index, indices in enumerate(model_indices):
        stats = manifest["model_stats"][model_index]
        require(
            len(indices) == args.expected_per_model_train_size,
            f"model {model_index} size={len(indices)}, expected={args.expected_per_model_train_size}",
        )
        require(
            int(stats["size"]) == args.expected_per_model_train_size,
            f"model_stats {model_index} size={stats['size']}, expected={args.expected_per_model_train_size}",
        )
        require(
            sum(int(value) for value in stats["class_counts"].values()) == args.expected_per_model_train_size,
            f"model_stats {model_index} class_counts do not sum to expected size",
        )
        require(
            len(set(indices)) == len(indices),
            f"model {model_index} contains duplicate sample indices",
        )
        require(
            set(indices).issubset(model_train),
            f"model {model_index} contains indices outside model_train",
        )

    expected_overlap = float(manifest["overlap_ratio"])
    actual_overlap = float(manifest["mean_pairwise_overlap"])
    require(
        abs(actual_overlap - expected_overlap) <= args.overlap_tolerance,
        f"mean_pairwise_overlap={actual_overlap}, expected={expected_overlap}",
    )

    print(f"validated split manifest: {run_dir}")
    print(f"mean_pairwise_overlap={actual_overlap:.6f}")


def main() -> None:
    validate(parse_args())


if __name__ == "__main__":
    main()
