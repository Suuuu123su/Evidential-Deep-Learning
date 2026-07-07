from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from edl_cifar5.diversity import metric_row, pairwise_diversity, per_class_accuracy
from edl_cifar5.fusion import apply_fusion_rule


MODEL_NAMES = ("lenet", "small_cnn", "small_vgg", "tiny_resnet", "tiny_vit")
METRIC_COLUMNS = ("accuracy", "nll", "brier", "ece")
DIVERSITY_COLUMNS = (
    "mean_disagreement",
    "mean_double_fault",
    "mean_same_wrong_prediction",
    "mean_correctness_corr",
    "mean_prob_corr",
    "mean_plausibility_corr",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select r on fusion_val and report final r-out-of-K metrics on test.")
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--accuracy-tie-tolerance", type=float, default=0.002)
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument("--make-plots", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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
        writer.writerows(rows)


def iter_run_dirs(root: Path) -> list[Path]:
    if (root / "fusion_val_outputs.npz").is_file() and (root / "test_outputs.npz").is_file():
        return [root]
    run_dirs = []
    for fusion_path in root.rglob("fusion_val_outputs.npz"):
        run_dir = fusion_path.parent
        if (run_dir / "test_outputs.npz").is_file():
            run_dirs.append(run_dir)
    return sorted(set(run_dirs))


def condition_metadata(run_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_dir": str(run_dir),
        "condition": run_dir.name,
        "partition_mode": manifest.get("partition_mode") or "unknown",
        "overlap_ratio": manifest.get("overlap_ratio"),
        "per_model_train_size": manifest.get("per_model_train_size"),
        "seed": manifest.get("seed"),
        "parallel_models": manifest.get("parallel_models"),
        "batch_size": manifest.get("batch_size"),
        "amp": manifest.get("amp"),
        "peak_vram_mb": manifest.get("peak_vram_mb"),
    }


def select_best_r(rows: list[dict[str, Any]], tolerance: float) -> dict[str, Any]:
    max_acc = max(float(row["accuracy"]) for row in rows)
    candidates = [row for row in rows if max_acc - float(row["accuracy"]) <= tolerance]
    return sorted(candidates, key=lambda row: (float(row["nll"]), float(row["ece"]), int(row["r"])))[0]


def fused_rows(
    base: dict[str, Any],
    split: str,
    prob: np.ndarray,
    plausibility: np.ndarray,
    labels: np.ndarray,
    bins: int,
) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    rows: list[dict[str, Any]] = []
    fused_by_r: list[np.ndarray] = []
    for r in range(1, prob.shape[1] + 1):
        fused = apply_fusion_rule("r_out_of_k", prob, plausibility, r=r)
        fused_by_r.append(fused)
        row = dict(base)
        row["split"] = split
        row["r"] = r
        row.update(metric_row(fused, labels, bins=bins))
        rows.append(row)
    return rows, fused_by_r


def model_metric_rows(base: dict[str, Any], prob: np.ndarray, labels: np.ndarray, bins: int) -> list[dict[str, Any]]:
    rows = []
    for model_index in range(prob.shape[1]):
        row = dict(base)
        row["split"] = "test"
        row["model_index"] = model_index
        row["model"] = MODEL_NAMES[model_index] if model_index < len(MODEL_NAMES) else f"model_{model_index}"
        row.update(metric_row(prob[:, model_index, :], labels, bins=bins))
        rows.append(row)
    return rows


def per_class_selected_rows(
    base: dict[str, Any],
    fusion_rows: list[dict[str, Any]],
    fusion_fused_by_r: list[np.ndarray],
    fusion_labels: np.ndarray,
    test_fused_by_r: list[np.ndarray],
    test_labels: np.ndarray,
    class_names: list[str],
    tolerance: float,
) -> list[dict[str, Any]]:
    output = []
    for class_id, class_name in enumerate(class_names):
        fusion_acc_by_r = []
        for r_index, fused in enumerate(fusion_fused_by_r):
            class_acc = per_class_accuracy(fused, fusion_labels, len(class_names))[class_id]
            fusion_acc_by_r.append((r_index + 1, class_acc))
        max_acc = max(acc for _, acc in fusion_acc_by_r)
        candidate_rs = [r for r, acc in fusion_acc_by_r if max_acc - acc <= tolerance]
        chosen_r = sorted(candidate_rs, key=lambda r: (float(fusion_rows[r - 1]["nll"]), float(fusion_rows[r - 1]["ece"]), r))[0]
        test_class_acc = per_class_accuracy(test_fused_by_r[chosen_r - 1], test_labels, len(class_names))[class_id]
        row = dict(base)
        row.update(
            {
                "class_id": class_id,
                "class_name": class_name,
                "selected_r": chosen_r,
                "fusion_val_class_accuracy": dict(fusion_acc_by_r)[chosen_r],
                "test_class_accuracy": test_class_acc,
            }
        )
        output.append(row)
    return output


def analyze_run(run_dir: Path, tie_tolerance: float, ece_bins: int) -> dict[str, list[dict[str, Any]]]:
    manifest = read_json(run_dir / "manifest.json")
    metadata = condition_metadata(run_dir, manifest)
    fusion_data = np.load(run_dir / "fusion_val_outputs.npz", allow_pickle=True)
    test_data = np.load(run_dir / "test_outputs.npz", allow_pickle=True)

    fusion_prob = fusion_data["prob"]
    fusion_plausibility = fusion_data["plausibility"]
    fusion_labels = fusion_data["labels"].astype(np.int64)
    test_prob = test_data["prob"]
    test_plausibility = test_data["plausibility"]
    test_labels = test_data["labels"].astype(np.int64)
    class_names = [str(x) for x in test_data["class_names"].tolist()]

    fusion_rows, fusion_fused_by_r = fused_rows(
        metadata, "fusion_val", fusion_prob, fusion_plausibility, fusion_labels, ece_bins
    )
    test_rows, test_fused_by_r = fused_rows(metadata, "test", test_prob, test_plausibility, test_labels, ece_bins)

    selected = select_best_r(fusion_rows, tie_tolerance)
    selected_r = int(selected["r"])
    selected_row = dict(metadata)
    selected_row.update(
        {
            "selection_split": "fusion_val",
            "selected_r": selected_r,
            "fusion_val_accuracy": selected["accuracy"],
            "fusion_val_nll": selected["nll"],
            "fusion_val_brier": selected["brier"],
            "fusion_val_ece": selected["ece"],
            "selection_rule": f"max fusion_val accuracy within {tie_tolerance}, then min NLL, then min ECE",
        }
    )

    selected_test = dict(test_rows[selected_r - 1])
    selected_test["selection_split"] = "fusion_val"
    selected_test["selected_r"] = selected_r
    selected_test["is_selected_r"] = 1

    oracle = dict(select_best_r(test_rows, tie_tolerance))
    oracle["selection_split"] = "test_oracle_diagnostic_only"
    oracle["selected_r"] = oracle["r"]
    oracle["selection_rule"] = "diagnostic only; do not use as formal result"

    diversity = pairwise_diversity(test_prob, test_plausibility, test_labels)
    diversity_row = dict(metadata)
    diversity_row["split"] = "test"
    for key, value in diversity.items():
        if key != "pairs":
            diversity_row[key] = value

    pair_rows = []
    for pair in diversity["pairs"]:
        row = dict(metadata)
        row["split"] = "test"
        row.update(pair)
        pair_rows.append(row)

    return {
        "fusion_val_routofk": fusion_rows,
        "test_routofk": test_rows,
        "selected_r_by_fusion_val": [selected_row],
        "test_metrics_by_selected_r": [selected_test],
        "oracle_test_best_r": [oracle],
        "per_class_selected_r": per_class_selected_rows(
            metadata,
            fusion_rows,
            fusion_fused_by_r,
            fusion_labels,
            test_fused_by_r,
            test_labels,
            class_names,
            tie_tolerance,
        ),
        "model_metrics_test": model_metric_rows(metadata, test_prob, test_labels, ece_bins),
        "diversity_summary": [diversity_row],
        "pairwise_diversity": pair_rows,
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((value - m) ** 2 for value in values) / (len(values) - 1))


def aggregate_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["partition_mode"]), float(row["overlap_ratio"])), []).append(row)
    output = []
    for (mode, overlap), group_rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        row: dict[str, Any] = {
            "partition_mode": mode,
            "overlap_ratio": overlap,
            "n": len(group_rows),
            "seeds": ",".join(str(int(float(r["seed"]))) for r in sorted(group_rows, key=lambda r: int(float(r["seed"])))),
        }
        for metric in METRIC_COLUMNS:
            values = [float(r[metric]) for r in group_rows]
            std = sample_std(values)
            row[f"{metric}_mean"] = mean(values)
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95"] = 1.96 * std / math.sqrt(len(values)) if values else math.nan
        output.append(row)
    return output


def selected_r_frequency(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float], list[int]] = {}
    for row in rows:
        grouped.setdefault((str(row["partition_mode"]), float(row["overlap_ratio"])), []).append(int(row["selected_r"]))
    output = []
    for (mode, overlap), selected_rs in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        for r in range(1, 6):
            output.append(
                {
                    "partition_mode": mode,
                    "overlap_ratio": overlap,
                    "selected_r": r,
                    "count": sum(1 for value in selected_rs if value == r),
                    "n": len(selected_rs),
                    "share": sum(1 for value in selected_rs if value == r) / len(selected_rs),
                }
            )
    return output


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(ys) < 2:
        return math.nan
    mx = mean(xs)
    my = mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0.0 or vy <= 0.0:
        return math.nan
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def diversity_correlations(diversity_rows: list[dict[str, Any]], selected_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected_by_condition = {row["condition"]: row for row in selected_rows}
    joined = []
    for row in diversity_rows:
        selected = selected_by_condition.get(row["condition"])
        if selected is None:
            continue
        merged = dict(row)
        merged["selected_r"] = selected["selected_r"]
        merged["test_accuracy"] = selected["accuracy"]
        joined.append(merged)
    output = []
    for metric in DIVERSITY_COLUMNS:
        xs = [float(row[metric]) for row in joined]
        output.append(
            {
                "diversity_metric": metric,
                "pearson_with_selected_r": pearson(xs, [float(row["selected_r"]) for row in joined]),
                "pearson_with_test_accuracy": pearson(xs, [float(row["test_accuracy"]) for row in joined]),
                "n": len(joined),
            }
        )
    return output


def maybe_make_plots(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipped plot generation.")
        return

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for mode in sorted({str(row["partition_mode"]) for row in rows}):
        mode_rows = sorted(
            [row for row in rows if str(row["partition_mode"]) == mode],
            key=lambda row: float(row["overlap_ratio"]),
        )
        plt.figure(figsize=(6.75, 3.0))
        plt.plot(
            [float(row["overlap_ratio"]) for row in mode_rows],
            [float(row["accuracy"]) for row in mode_rows],
            marker="o",
        )
        plt.xlabel("Training-data overlap ratio")
        plt.ylabel("Test accuracy selected by fusion_val")
        plt.title(mode)
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(figure_dir / f"selected_test_accuracy_{mode}.png", dpi=300)
        plt.savefig(figure_dir / f"selected_test_accuracy_{mode}.pdf")
        plt.close()


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root)
    output_dir = Path(args.output_dir) if args.output_dir else runs_root / "analysis"
    run_dirs = iter_run_dirs(runs_root)
    if not run_dirs:
        raise FileNotFoundError(f"No paired fusion_val_outputs.npz and test_outputs.npz files found under {runs_root}")

    combined: dict[str, list[dict[str, Any]]] = {
        "fusion_val_routofk": [],
        "test_routofk": [],
        "selected_r_by_fusion_val": [],
        "test_metrics_by_selected_r": [],
        "oracle_test_best_r": [],
        "per_class_selected_r": [],
        "model_metrics_test": [],
        "diversity_summary": [],
        "pairwise_diversity": [],
    }
    for run_dir in run_dirs:
        result = analyze_run(run_dir, args.accuracy_tie_tolerance, args.ece_bins)
        for key, rows in result.items():
            combined[key].extend(rows)

    combined["aggregate_test_metrics_by_condition"] = aggregate_metrics(combined["test_metrics_by_selected_r"])
    combined["selected_r_frequency"] = selected_r_frequency(combined["selected_r_by_fusion_val"])
    combined["diversity_metric_correlations"] = diversity_correlations(
        combined["diversity_summary"],
        combined["test_metrics_by_selected_r"],
    )

    for name, rows in combined.items():
        write_csv(output_dir / f"{name}.csv", rows)

    if args.make_plots:
        maybe_make_plots(output_dir, combined["test_metrics_by_selected_r"])

    print(f"Analyzed {len(run_dirs)} run(s) with fusion_val selection.")
    print(f"Wrote analysis outputs to {output_dir}")


if __name__ == "__main__":
    main()
