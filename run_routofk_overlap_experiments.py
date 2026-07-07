from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
import time


OVERLAPS = ("1.0", "0.75", "0.5", "0.25", "0.0")
PARTITION_MODES = ("stratified-balanced", "class-skewed-private")
SEEDS = ("2026", "2027", "2028")


@dataclass
class ActiveJob:
    condition: str
    process: subprocess.Popen
    stdout_file: object
    stderr_file: object
    started_at: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the r-out-of-K overlap experiment grid.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--data-source", choices=["auto", "torchvision", "imagefolder"], default="imagefolder")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--parallel-models", type=int, default=2)
    parser.add_argument("--max-concurrent-runs", type=int, default=2)
    parser.add_argument("--min-free-vram-mb", type=int, default=2500)
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", default=",".join(SEEDS))
    parser.add_argument("--per-model-train-size", type=int, default=4000)
    parser.add_argument("--model-train-per-class", type=int, default=4000)
    parser.add_argument("--checkpoint-val-per-class", type=int, default=500)
    parser.add_argument("--fusion-val-per-class", type=int, default=500)
    parser.add_argument("--overlaps", default=",".join(OVERLAPS))
    parser.add_argument("--partition-modes", default=",".join(PARTITION_MODES))
    parser.add_argument("--amp", choices=["auto", "none", "fp16", "bf16"], default="auto")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--disable-progress", action="store_true")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-split-validation", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def condition_name(mode: str, overlap: str, seed: int) -> str:
    return f"{mode}_overlap_{overlap.replace('.', '')}_seed{seed}"


def parse_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def free_vram_mib() -> int | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    first = result.stdout.strip().splitlines()[0].strip()
    try:
        return int(first)
    except ValueError:
        return None


def build_condition_command(args: argparse.Namespace, run_dir: Path, mode: str, overlap: str, seed: int) -> list[str]:
    project_root = Path(__file__).resolve().parent
    cmd = [
        sys.executable,
        str(project_root / "run_routofk_condition.py"),
        "--data-dir",
        args.data_dir,
        "--data-source",
        args.data_source,
        "--run-dir",
        str(run_dir),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--parallel-models",
        str(args.parallel_models),
        "--device",
        args.device,
        "--seed",
        str(seed),
        "--overlap-ratio",
        overlap,
        "--partition-mode",
        mode,
        "--per-model-train-size",
        str(args.per_model_train_size),
        "--model-train-per-class",
        str(args.model_train_per_class),
        "--checkpoint-val-per-class",
        str(args.checkpoint_val_per_class),
        "--fusion-val-per-class",
        str(args.fusion_val_per_class),
        "--amp",
        args.amp,
        "--prefetch-factor",
        str(args.prefetch_factor),
    ]
    if args.compile:
        cmd.append("--compile")
    if args.disable_progress:
        cmd.append("--disable-progress")
    if args.skip_existing:
        cmd.append("--skip-existing")
    if args.dry_run:
        cmd.append("--dry-run")
    return cmd


def close_job(job: ActiveJob) -> None:
    job.stdout_file.close()
    job.stderr_file.close()


def wait_for_one(active: list[ActiveJob], failures: list[str], poll_seconds: int) -> None:
    while True:
        for job in list(active):
            return_code = job.process.poll()
            if return_code is None:
                continue
            elapsed = time.time() - job.started_at
            active.remove(job)
            close_job(job)
            if return_code != 0:
                failures.append(f"{job.condition} failed with exit code {return_code}")
                print(f"FAILED {job.condition} after {elapsed:.1f}s", flush=True)
            else:
                print(f"finished {job.condition} in {elapsed:.1f}s", flush=True)
            return
        time.sleep(poll_seconds)


def launch_job(cmd: list[str], run_dir: Path, condition: str) -> ActiveJob:
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_file = (run_dir / "condition_stdout.log").open("w", encoding="utf-8")
    stderr_file = (run_dir / "condition_stderr.log").open("w", encoding="utf-8")
    print(" ".join(cmd), flush=True)
    process = subprocess.Popen(cmd, stdout=stdout_file, stderr=stderr_file)
    return ActiveJob(condition, process, stdout_file, stderr_file, time.time())


def validate_splits(args: argparse.Namespace, run_dirs: list[Path]) -> None:
    project_root = Path(__file__).resolve().parent
    for run_dir in run_dirs:
        split_path = run_dir / "splits.json"
        if not split_path.is_file():
            raise FileNotFoundError(f"Missing split manifest after condition run: {split_path}")
        cmd = [
            sys.executable,
            str(project_root / "validate_routofk_splits.py"),
            "--run-dir",
            str(run_dir),
            "--expected-model-train-per-class",
            str(args.model_train_per_class),
            "--expected-checkpoint-val-per-class",
            str(args.checkpoint_val_per_class),
            "--expected-fusion-val-per-class",
            str(args.fusion_val_per_class),
            "--expected-per-model-train-size",
            str(args.per_model_train_size),
        ]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    overlaps = parse_list(args.overlaps)
    partition_modes = parse_list(args.partition_modes)
    seeds = [int(seed) for seed in parse_list(args.seeds)]

    jobs: list[tuple[str, Path, list[str]]] = []
    expected_run_dirs: list[Path] = []
    for seed in seeds:
        for mode in partition_modes:
            if mode not in PARTITION_MODES:
                raise ValueError(f"Unknown partition mode: {mode}")
            for overlap in overlaps:
                float(overlap)
                condition = condition_name(mode, overlap, seed)
                run_dir = output_root / condition
                expected_run_dirs.append(run_dir)
                if args.skip_existing and (run_dir / "fusion_val_outputs.npz").is_file() and (run_dir / "test_outputs.npz").is_file():
                    print(f"skip existing: {run_dir}", flush=True)
                    continue
                jobs.append((condition, run_dir, build_condition_command(args, run_dir, mode, overlap, seed)))

    active: list[ActiveJob] = []
    failures: list[str] = []
    for condition, run_dir, cmd in jobs:
        while len(active) >= args.max_concurrent_runs:
            wait_for_one(active, failures, args.poll_seconds)
        if args.device.startswith("cuda") and args.min_free_vram_mb > 0:
            while True:
                free = free_vram_mib()
                if free is None or free >= args.min_free_vram_mb:
                    break
                print(f"waiting for free VRAM: {free} MiB < {args.min_free_vram_mb} MiB", flush=True)
                wait_for_one(active, failures, args.poll_seconds) if active else time.sleep(args.poll_seconds)
        if args.dry_run:
            print(" ".join(cmd), flush=True)
        else:
            active.append(launch_job(cmd, run_dir, condition))

    while active:
        wait_for_one(active, failures, args.poll_seconds)

    if failures:
        raise RuntimeError("; ".join(failures))

    if not args.dry_run and not args.skip_split_validation:
        validate_splits(args, expected_run_dirs)

    analysis_cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "analyze_routofk_fusionval.py"),
        "--runs-root",
        str(output_root),
        "--make-plots",
    ]
    print(" ".join(analysis_cmd), flush=True)
    if not args.dry_run:
        subprocess.run(analysis_cmd, check=True)


if __name__ == "__main__":
    main()
