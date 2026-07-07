from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and export one r-out-of-K overlap condition.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data-source", choices=["auto", "torchvision", "imagefolder"], default="imagefolder")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--parallel-models", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overlap-ratio", required=True)
    parser.add_argument("--partition-mode", choices=["stratified-balanced", "class-skewed-private"], required=True)
    parser.add_argument("--per-model-train-size", type=int, default=4000)
    parser.add_argument("--model-train-per-class", type=int, default=4000)
    parser.add_argument("--checkpoint-val-per-class", type=int, default=500)
    parser.add_argument("--fusion-val-per-class", type=int, default=500)
    parser.add_argument("--amp", choices=["auto", "none", "fp16", "bf16"], default="auto")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--disable-progress", action="store_true")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_command(cmd: list[str], *, dry_run: bool) -> None:
    print(" ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    run_dir = Path(args.run_dir)
    fusion_outputs = run_dir / "fusion_val_outputs.npz"
    test_outputs = run_dir / "test_outputs.npz"
    if args.skip_existing and fusion_outputs.is_file() and test_outputs.is_file():
        print(f"skip existing condition: {run_dir}", flush=True)
        return

    train_cmd = [
        sys.executable,
        str(project_root / "train_cifar5_edl.py"),
        "--data-dir",
        args.data_dir,
        "--data-source",
        args.data_source,
        "--no-download",
        "--output-dir",
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
        str(args.seed),
        "--per-model-splits",
        "--overlap-ratio",
        str(args.overlap_ratio),
        "--partition-mode",
        args.partition_mode,
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
        train_cmd.append("--compile")
    if args.disable_progress:
        train_cmd.append("--disable-progress")

    export_common = [
        sys.executable,
        str(project_root / "export_outputs.py"),
        "--data-dir",
        args.data_dir,
        "--data-source",
        args.data_source,
        "--no-download",
        "--checkpoint-dir",
        str(run_dir),
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--amp",
        args.amp,
        "--prefetch-factor",
        str(args.prefetch_factor),
    ]
    export_fusion_cmd = export_common + ["--split", "fusion_val", "--output", str(fusion_outputs)]
    export_test_cmd = export_common + ["--split", "test", "--output", str(test_outputs)]

    run_command(train_cmd, dry_run=args.dry_run)
    run_command(export_fusion_cmd, dry_run=args.dry_run)
    run_command(export_test_cmd, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
