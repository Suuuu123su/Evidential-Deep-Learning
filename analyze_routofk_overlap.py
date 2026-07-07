from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from edl_cifar5.diversity import metric_row, pairwise_diversity, per_class_accuracy
from edl_cifar5.fusion import apply_fusion_rule


MODEL_NAMES = ("lenet", "small_cnn", "small_vgg", "tiny_resnet", "tiny_vit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze r-out-of-K under training-data overlap conditions.")
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--accuracy-tie-tolerance", type=float, default=0.002)
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument("--make-plots", action="store_true")
    return parser.parse_args()


def read_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "manifest.json"
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_run_dirs(root: Path) -> list[Path]:
    if (root / "test_outputs.npz").is_file():
        return [root]
    run_dirs = [path for path in root.rglob("test_outputs.npz")]
    return sorted({path.parent for path in run_dirs})


def condition_metadata(run_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_dir": str(run_dir),
        "condition": run_dir.name,
        "partition_mode": manifest.get("partition_mode") or "unknown",
        "overlap_ratio": manifest.get("overlap_ratio"),
        "per_model_train_size": manifest.get("per_model_train_size"),
        "seed": manifest.get("seed"),
        "parallel_models": manifest.get("parallel_models"),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def select_best_r(rows: list[dict[str, Any]], tolerance: float) -> dict[str, Any]:
    max_acc = max(float(row["accuracy"]) for row in rows)
    candidates = [row for row in rows if max_acc - float(row["accuracy"]) <= tolerance]
    return sorted(candidates, key=lambda row: (float(row["nll"]), float(row["ece"]), int(row["r"])))[0]


def add_class_metrics(base: dict[str, Any], prob: np.ndarray, labels: np.ndarray, class_names: list[str]) -> list[dict[str, Any]]:
    rows = []
    for class_id, acc in per_class_accuracy(prob, labels, len(class_names)).items():
        row = dict(base)
        row["class_id"] = class_id
        row["class_name"] = class_names[class_id]
        row["class_accuracy"] = acc
        rows.append(row)
    return rows


def analyze_run(run_dir: Path, tie_tolerance: float, ece_bins: int) -> dict[str, list[dict[str, Any]]]:
    manifest = read_manifest(run_dir)
    metadata = condition_metadata(run_dir, manifest)
    data = np.load(run_dir / "test_outputs.npz", allow_pickle=True)
    prob = data["prob"]
    plausibility = data["plausibility"]
    labels = data["labels"].astype(np.int64)
    class_names = [str(x) for x in data["class_names"].tolist()]
    source_count = prob.shape[1]

    model_rows: list[dict[str, Any]] = []
    for model_index in range(source_count):
        row = dict(metadata)
        row["model_index"] = model_index
        row["model"] = MODEL_NAMES[model_index] if model_index < len(MODEL_NAMES) else f"model_{model_index}"
        row.update(metric_row(prob[:, model_index, :], labels, bins=ece_bins))
        model_rows.append(row)

    r_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    for r in range(1, source_count + 1):
        fused = apply_fusion_rule("r_out_of_k", prob, plausibility, r=r)
        row = dict(metadata)
        row["r"] = r
        row.update(metric_row(fused, labels, bins=ece_bins))
        r_rows.append(row)
        class_rows.extend(add_class_metrics(row, fused, labels, class_names))

    best = select_best_r(r_rows, tie_tolerance)
    best_rows = []
    for row in r_rows:
        row["is_best_r"] = int(int(row["r"]) == int(best["r"]))
    best_row = dict(best)
    best_row["selection_rule"] = f"max accuracy within {tie_tolerance}, then min NLL, then min ECE"
    best_rows.append(best_row)

    per_class_best: list[dict[str, Any]] = []
    for class_id, class_name in enumerate(class_names):
        candidates = [row for row in class_rows if int(row["class_id"]) == class_id]
        max_acc = max(float(row["class_accuracy"]) for row in candidates)
        close = [row for row in candidates if max_acc - float(row["class_accuracy"]) <= tie_tolerance]
        chosen = sorted(close, key=lambda row: (float(row["nll"]), float(row["ece"]), int(row["r"])))[0]
        out = {
            "run_dir": metadata["run_dir"],
            "condition": metadata["condition"],
            "partition_mode": metadata["partition_mode"],
            "overlap_ratio": metadata["overlap_ratio"],
            "class_id": class_id,
            "class_name": class_name,
            "best_r": chosen["r"],
            "class_accuracy": chosen["class_accuracy"],
        }
        per_class_best.append(out)

    diversity = pairwise_diversity(prob, plausibility, labels)
    diversity_row = dict(metadata)
    for key, value in diversity.items():
        if key != "pairs":
            diversity_row[key] = value

    pair_rows = []
    for row in diversity["pairs"]:
        out = dict(metadata)
        out.update(row)
        pair_rows.append(out)

    return {
        "summary_models": model_rows,
        "summary_routofk": r_rows,
        "best_r_by_condition": best_rows,
        "per_class_best_r": per_class_best,
        "summary_diversity": [diversity_row],
        "pairwise_diversity": pair_rows,
        "per_class_routofk": class_rows,
    }


def maybe_make_plots(output_dir: Path, summary_rows: list[dict[str, Any]], best_rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipped plot generation.")
        return

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    modes = sorted({str(row["partition_mode"]) for row in summary_rows})
    for mode in modes:
        mode_rows = [row for row in summary_rows if str(row["partition_mode"]) == mode and row["overlap_ratio"] != ""]
        if not mode_rows:
            continue
        overlaps = sorted({float(row["overlap_ratio"]) for row in mode_rows})
        rs = sorted({int(row["r"]) for row in mode_rows})

        plt.figure(figsize=(6.75, 3.0))
        for r in rs:
            values = []
            for overlap in overlaps:
                match = [row for row in mode_rows if int(row["r"]) == r and float(row["overlap_ratio"]) == overlap]
                values.append(float(match[0]["accuracy"]) if match else np.nan)
            plt.plot(overlaps, values, marker="o", label=f"r={r}")
        plt.xlabel("Training-data overlap ratio")
        plt.ylabel("Test accuracy")
        plt.title(f"r-out-of-K accuracy ({mode})")
        plt.grid(alpha=0.2)
        plt.legend(ncol=min(5, len(rs)))
        plt.tight_layout()
        plt.savefig(figure_dir / f"accuracy_by_r_{mode}.png", dpi=300)
        plt.savefig(figure_dir / f"accuracy_by_r_{mode}.pdf")
        plt.close()

    best_plot_rows = [row for row in best_rows if row["overlap_ratio"] != ""]
    if best_plot_rows:
        plt.figure(figsize=(6.75, 3.0))
        for mode in sorted({str(row["partition_mode"]) for row in best_plot_rows}):
            rows = sorted(
                [row for row in best_plot_rows if str(row["partition_mode"]) == mode],
                key=lambda row: float(row["overlap_ratio"]),
            )
            plt.plot(
                [float(row["overlap_ratio"]) for row in rows],
                [int(row["r"]) for row in rows],
                marker="o",
                label=mode,
            )
        plt.xlabel("Training-data overlap ratio")
        plt.ylabel("Selected best r")
        plt.yticks([1, 2, 3, 4, 5])
        plt.grid(alpha=0.2)
        plt.legend()
        plt.tight_layout()
        plt.savefig(figure_dir / "best_r_by_overlap.png", dpi=300)
        plt.savefig(figure_dir / "best_r_by_overlap.pdf")
        plt.close()


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root)
    output_dir = Path(args.output_dir) if args.output_dir else runs_root / "analysis"
    run_dirs = iter_run_dirs(runs_root)
    if not run_dirs:
        raise FileNotFoundError(f"No test_outputs.npz files found under {runs_root}")

    combined: dict[str, list[dict[str, Any]]] = {
        "summary_models": [],
        "summary_routofk": [],
        "best_r_by_condition": [],
        "per_class_best_r": [],
        "summary_diversity": [],
        "pairwise_diversity": [],
        "per_class_routofk": [],
    }
    for run_dir in run_dirs:
        result = analyze_run(run_dir, args.accuracy_tie_tolerance, args.ece_bins)
        for key, rows in result.items():
            combined[key].extend(rows)

    for name, rows in combined.items():
        write_csv(output_dir / f"{name}.csv", rows)

    if args.make_plots:
        maybe_make_plots(output_dir, combined["summary_routofk"], combined["best_r_by_condition"])

    print(f"Analyzed {len(run_dirs)} run(s).")
    print(f"Wrote analysis outputs to {output_dir}")


if __name__ == "__main__":
    main()
