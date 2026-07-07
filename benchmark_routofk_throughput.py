from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run small throughput benchmarks for r-out-of-K training.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--amp", choices=["auto", "none", "fp16", "bf16"], default="auto")
    parser.add_argument("--sample-seconds", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def query_gpu() -> dict[str, float] | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    first = result.stdout.strip().splitlines()[0].split(",")
    if len(first) < 3:
        return None
    return {
        "gpu_util_pct": safe_float(first[0].strip()),
        "memory_used_mib": safe_float(first[1].strip()),
        "memory_free_mib": safe_float(first[2].strip()),
    }


def summarize_histories(output_dir: Path) -> dict[str, float]:
    epoch_seconds = []
    train_images_per_sec = []
    aggregate_images_per_sec = []
    nan_detected = 0
    peak_vram_values = []
    for manifest_path in output_dir.rglob("manifest.json"):
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        if manifest.get("peak_vram_mb") is not None:
            peak_vram_values.append(safe_float(manifest.get("peak_vram_mb")))
    for history_path in output_dir.rglob("*_history.json"):
        with history_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        for row in payload.get("history", []):
            for key, value in row.items():
                if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                    nan_detected = 1
            if "epoch_seconds" in row:
                epoch_seconds.append(safe_float(row["epoch_seconds"]))
            if "train_images_per_sec" in row:
                train_images_per_sec.append(safe_float(row["train_images_per_sec"]))
            if "aggregate_images_per_sec" in row:
                aggregate_images_per_sec.append(safe_float(row["aggregate_images_per_sec"]))

    def finite_mean(values: list[float]) -> float:
        finite = [value for value in values if math.isfinite(value)]
        return sum(finite) / len(finite) if finite else math.nan

    finite_peak = [value for value in peak_vram_values if math.isfinite(value)]
    return {
        "mean_epoch_seconds": finite_mean(epoch_seconds),
        "mean_train_images_per_sec_from_history": finite_mean(train_images_per_sec),
        "mean_aggregate_images_per_sec_from_history": finite_mean(aggregate_images_per_sec),
        "peak_vram_mb_manifest": max(finite_peak) if finite_peak else math.nan,
        "nan_detected": nan_detected,
    }


def run_case(
    name: str,
    cmd: list[str],
    *,
    output_dir: Path,
    expected_train_images: int,
    sample_seconds: float,
    dry_run: bool,
) -> dict:
    print(f"benchmark {name}", flush=True)
    print(" ".join(cmd), flush=True)
    if dry_run:
        return {"name": name, "seconds": 0.0, "return_code": 0, "expected_train_images": expected_train_images}
    start = time.perf_counter()
    process = subprocess.Popen(cmd)
    samples = []
    next_sample = start
    while process.poll() is None:
        now = time.perf_counter()
        if now >= next_sample:
            sample = query_gpu()
            if sample is not None:
                samples.append(sample)
            next_sample = now + sample_seconds
        time.sleep(0.25)
    seconds = time.perf_counter() - start
    summary = summarize_histories(output_dir)
    util_values = [sample["gpu_util_pct"] for sample in samples if math.isfinite(sample["gpu_util_pct"])]
    mem_values = [sample["memory_used_mib"] for sample in samples if math.isfinite(sample["memory_used_mib"])]
    row = {
        "name": name,
        "seconds": seconds,
        "return_code": process.returncode,
        "expected_train_images": expected_train_images,
        "effective_train_images_per_sec": expected_train_images / seconds if seconds > 0 else math.nan,
        "gpu_util_avg_pct": sum(util_values) / len(util_values) if util_values else math.nan,
        "gpu_util_max_pct": max(util_values) if util_values else math.nan,
        "gpu_mem_peak_mib_sampled": max(mem_values) if mem_values else math.nan,
        "gpu_sample_count": len(samples),
    }
    row.update(summary)
    return row


def runner_cmd(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    seeds: str,
    batch_size: int,
    parallel_models: int,
    max_concurrent_runs: int,
) -> list[str]:
    project_root = Path(__file__).resolve().parent
    return [
        sys.executable,
        str(project_root / "run_routofk_overlap_experiments.py"),
        "--data-dir",
        args.data_dir,
        "--output-root",
        str(output_dir),
        "--epochs",
        "3",
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(args.num_workers),
        "--parallel-models",
        str(parallel_models),
        "--max-concurrent-runs",
        str(max_concurrent_runs),
        "--device",
        args.device,
        "--seeds",
        seeds,
        "--overlaps",
        "0.5",
        "--partition-modes",
        "stratified-balanced",
        "--amp",
        args.amp,
        "--disable-progress",
        "--skip-existing",
    ]


def expected_images(condition_count: int, batch_size: int, epochs: int = 3) -> int:
    images_per_model_epoch = (4000 // batch_size) * batch_size
    return condition_count * 5 * epochs * images_per_model_epoch


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    cases = [
        (
            "A_single_process_parallel5_bs512",
            output_root / "A_single_process_parallel5_bs512",
            expected_images(condition_count=1, batch_size=512),
            runner_cmd(
                args,
                output_dir=output_root / "A_single_process_parallel5_bs512",
                seeds="2026",
                batch_size=512,
                parallel_models=5,
                max_concurrent_runs=1,
            ),
        ),
        (
            "B_two_process_parallel2_bs512",
            output_root / "B_two_process_parallel2_bs512",
            expected_images(condition_count=2, batch_size=512),
            runner_cmd(
                args,
                output_dir=output_root / "B_two_process_parallel2_bs512",
                seeds="2026,2027",
                batch_size=512,
                parallel_models=2,
                max_concurrent_runs=2,
            ),
        ),
        (
            "C_two_process_parallel1_bs768",
            output_root / "C_two_process_parallel1_bs768",
            expected_images(condition_count=2, batch_size=768),
            runner_cmd(
                args,
                output_dir=output_root / "C_two_process_parallel1_bs768",
                seeds="2026,2027",
                batch_size=768,
                parallel_models=1,
                max_concurrent_runs=2,
            ),
        ),
    ]

    rows = []
    for name, output_dir, expected_train_images, cmd in cases:
        row = run_case(
            name,
            cmd,
            output_dir=output_dir,
            expected_train_images=expected_train_images,
            sample_seconds=args.sample_seconds,
            dry_run=args.dry_run,
        )
        rows.append(row)
        if row["return_code"] != 0:
            print(f"{name} failed; keeping logs for inspection and continuing.", flush=True)

    summary = output_root / "benchmark_summary.tsv"
    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
        with summary.open("w", encoding="utf-8") as f:
            columns = list(rows[0].keys()) if rows else ["name", "seconds", "return_code"]
            f.write("\t".join(columns) + "\n")
            for row in rows:
                f.write("\t".join(str(row.get(column, "")) for column in columns) + "\n")
        print(f"wrote {summary}", flush=True)


if __name__ == "__main__":
    main()
