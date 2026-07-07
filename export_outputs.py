from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from edl_cifar5.data import Cifar5Spec, build_cifar5_dataset, parse_classes
from edl_cifar5.evidence import alpha_to_mass, alpha_to_plausibility, alpha_to_prob
from edl_cifar5.models import MODEL_REGISTRY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export alpha, mass, and plausibility from trained EDL classifiers.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=["train", "test", "model_train", "checkpoint_val", "fusion_val"], default="test")
    parser.add_argument("--classes", default="0,1,2,3,4")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--subset-size", type=int, default=None)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--data-source", choices=["auto", "torchvision", "imagefolder"], default="auto")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--no-persistent-workers", action="store_true")
    parser.add_argument("--amp", choices=["auto", "none", "fp16", "bf16"], default="auto")
    return parser.parse_args()


def load_models(checkpoint_dir: Path, device: torch.device) -> list[torch.nn.Module]:
    checkpoints = sorted(checkpoint_dir.glob("*.pt"))
    if len(checkpoints) != 5:
        raise ValueError(f"Expected exactly 5 checkpoints in {checkpoint_dir}, found {len(checkpoints)}")
    models = []
    for ckpt_path in checkpoints:
        ckpt = torch.load(ckpt_path, map_location=device)
        model_name = ckpt["model_name"]
        model = MODEL_REGISTRY[model_name](num_classes=5).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        models.append(model)
    return models


def split_indices_from_manifest(checkpoint_dir: Path, split: str) -> list[int]:
    path = checkpoint_dir / "splits.json"
    if not path.is_file():
        raise FileNotFoundError(f"{split} export requires split manifest: {path}")
    with path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    key_by_split = {
        "model_train": "model_train_indices",
        "checkpoint_val": "checkpoint_val_indices",
        "fusion_val": "fusion_val_indices",
    }
    key = key_by_split[split]
    if key not in manifest:
        raise KeyError(f"Split manifest missing {key}; cannot export {split}")
    return [int(idx) for idx in manifest[key]]


def build_export_dataset(args: argparse.Namespace, selected_classes: tuple[int, ...], checkpoint_dir: Path):
    if args.split in {"model_train", "checkpoint_val", "fusion_val"}:
        base = build_cifar5_dataset(
            args.data_dir,
            train=True,
            classes=selected_classes,
            download=not args.no_download,
            subset_size=args.subset_size,
            augment=False,
            source=args.data_source,
        )
        return Subset(base, split_indices_from_manifest(checkpoint_dir, args.split))
    return build_cifar5_dataset(
        args.data_dir,
        train=args.split == "train",
        classes=selected_classes,
        download=not args.no_download,
        subset_size=args.subset_size,
        augment=False,
        source=args.data_source,
    )


def loader_kwargs(args: argparse.Namespace) -> dict:
    kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": True,
    }
    if args.num_workers > 0:
        kwargs["persistent_workers"] = not args.no_persistent_workers
        kwargs["prefetch_factor"] = args.prefetch_factor
    return kwargs


def resolve_amp(args: argparse.Namespace, device: torch.device) -> tuple[bool, torch.dtype | None]:
    if device.type != "cuda" or args.amp == "none":
        return False, None
    if args.amp == "bf16":
        return True, torch.bfloat16
    if args.amp == "fp16":
        return True, torch.float16
    return True, torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


@torch.no_grad()
def main() -> None:
    args = parse_args()
    selected_classes = parse_classes(args.classes)
    spec = Cifar5Spec(selected_classes)
    device = torch.device(args.device)
    checkpoint_dir = Path(args.checkpoint_dir)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
    dataset = build_export_dataset(args, selected_classes, checkpoint_dir)
    loader = DataLoader(dataset, **loader_kwargs(args))
    models = load_models(checkpoint_dir, device)
    amp_enabled, amp_dtype = resolve_amp(args, device)

    all_alpha = []
    all_labels = []
    for images, labels in tqdm(loader, desc="export"):
        images = images.to(device, non_blocking=device.type == "cuda")
        batch_alpha = []
        for model in models:
            if amp_enabled and amp_dtype is not None:
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
                    out = model(images)
            else:
                out = model(images)
            batch_alpha.append(out["alpha"].detach().cpu().numpy())
        all_alpha.append(np.stack(batch_alpha, axis=1))
        all_labels.append(labels.numpy())

    alpha = np.concatenate(all_alpha, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    prob = alpha_to_prob(alpha)
    belief, uncertainty, plausibility = alpha_to_plausibility(alpha)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        alpha=alpha,
        prob=prob,
        belief=belief,
        uncertainty=uncertainty,
        plausibility=plausibility,
        labels=labels,
        classes=np.asarray(selected_classes, dtype=np.int64),
        class_names=np.asarray(spec.names),
        split=np.asarray(args.split),
    )
    print(f"wrote {output_path}")
    print(f"alpha shape: {alpha.shape}")


if __name__ == "__main__":
    main()
