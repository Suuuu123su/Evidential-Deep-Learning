from __future__ import annotations

import argparse
import contextlib
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from edl_cifar5.data import Cifar5Spec, build_cifar5_dataset, parse_classes, seed_worker
from edl_cifar5.data_splits import (
    build_per_model_splits,
    save_split_manifest,
    stratified_three_way_indices,
    stratified_train_val_indices,
)
from edl_cifar5.models import MODEL_REGISTRY
from edl_cifar5.train_utils import accuracy_from_prob, edl_loss, save_json, set_seed


DEFAULT_MODELS = ("lenet", "small_cnn", "small_vgg", "tiny_resnet", "tiny_vit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train five EDL classifiers on CIFAR-5.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="runs/cifar5_edl")
    parser.add_argument("--classes", default="0,1,2,3,4")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--annealing-epochs", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--parallel-models", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--subset-size", type=int, default=None)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--data-source", choices=["auto", "torchvision", "imagefolder"], default="auto")
    parser.add_argument("--per-model-splits", action="store_true")
    parser.add_argument("--overlap-ratio", type=float, default=1.0)
    parser.add_argument(
        "--partition-mode",
        choices=["stratified-balanced", "class-skewed-private"],
        default="stratified-balanced",
    )
    parser.add_argument("--per-model-train-size", type=int, default=4000)
    parser.add_argument("--model-train-per-class", type=int, default=4000)
    parser.add_argument("--checkpoint-val-per-class", type=int, default=500)
    parser.add_argument("--fusion-val-per-class", type=int, default=500)
    parser.add_argument("--amp", choices=["auto", "none", "fp16", "bf16"], default="auto")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--disable-progress", action="store_true")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--no-persistent-workers", action="store_true")
    parser.add_argument("--no-drop-last", action="store_true")
    return parser.parse_args()


def chunked(items: list[tuple[int, str]], size: int) -> list[list[tuple[int, str]]]:
    return [items[start : start + size] for start in range(0, len(items), size)]


def configure_torch(device: torch.device) -> None:
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")


def make_grad_scaler(device: torch.device, enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler(device.type, enabled=enabled)
        except TypeError:
            pass
    return torch.cuda.amp.GradScaler(enabled=enabled)


def resolve_amp(args: argparse.Namespace, device: torch.device):
    if device.type != "cuda" or args.amp == "none":
        return False, None, make_grad_scaler(device, enabled=False)
    if args.amp == "bf16":
        dtype = torch.bfloat16
    elif args.amp == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return True, dtype, make_grad_scaler(device, enabled=dtype == torch.float16)


def autocast_context(enabled: bool, dtype: torch.dtype | None, device: torch.device):
    if not enabled or dtype is None:
        return contextlib.nullcontext()
    return torch.amp.autocast(device_type=device.type, dtype=dtype)


def move_batch(images: torch.Tensor, labels: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    non_blocking = device.type == "cuda"
    return images.to(device, non_blocking=non_blocking), labels.to(device, non_blocking=non_blocking)


def maybe_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def progress_iter(iterable, args: argparse.Namespace, **kwargs):
    if args.disable_progress:
        return iterable
    return tqdm(iterable, **kwargs)


def set_progress_postfix(progress, args: argparse.Namespace, **kwargs) -> None:
    if not args.disable_progress:
        progress.set_postfix(**kwargs)


def state_dict_for_save(model: torch.nn.Module) -> dict:
    if hasattr(model, "_orig_mod"):
        return model._orig_mod.state_dict()
    return model.state_dict()


def dataloader_kwargs(args: argparse.Namespace, *, shuffle: bool, train: bool) -> dict:
    kwargs = {
        "batch_size": args.batch_size,
        "shuffle": shuffle,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "worker_init_fn": seed_worker,
        "drop_last": train and not args.no_drop_last,
    }
    if args.num_workers > 0:
        kwargs["persistent_workers"] = not args.no_persistent_workers
        kwargs["prefetch_factor"] = args.prefetch_factor
    return kwargs


def make_train_loader(args: argparse.Namespace, dataset: Subset, *, shuffle: bool = True) -> DataLoader:
    return DataLoader(dataset, **dataloader_kwargs(args, shuffle=shuffle, train=True))


def make_val_loader(args: argparse.Namespace, dataset: Subset) -> DataLoader:
    return DataLoader(dataset, **dataloader_kwargs(args, shuffle=False, train=False))


def train_one_model(
    model_name: str,
    model_index: int,
    args: argparse.Namespace,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> dict:
    set_seed(args.seed + model_index)
    amp_enabled, amp_dtype, scaler = resolve_amp(args, device)
    model = MODEL_REGISTRY[model_name](num_classes=num_classes).to(device)
    if args.compile:
        model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history: list[dict] = []
    best_val_acc = -1.0
    best_path = Path(args.output_dir) / f"{model_index:02d}_{model_name}.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        maybe_sync(device)
        epoch_start = time.perf_counter()
        train_loss_sum = 0.0
        train_acc_sum = 0.0
        train_count = 0
        progress = progress_iter(train_loader, args, desc=f"{model_name} epoch {epoch}/{args.epochs}", leave=False)
        for images, labels in progress:
            images, labels = move_batch(images, labels, device)
            with autocast_context(amp_enabled, amp_dtype, device):
                output = model(images)
            loss = edl_loss(output["alpha"], labels, epoch, num_classes, args.annealing_epochs)
            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            batch_size = labels.shape[0]
            train_loss_sum += float(loss.item()) * batch_size
            train_acc_sum += accuracy_from_prob(output["prob"].detach(), labels) * batch_size
            train_count += batch_size
            set_progress_postfix(progress, args, loss=train_loss_sum / train_count, acc=train_acc_sum / train_count)

        maybe_sync(device)
        epoch_seconds = time.perf_counter() - epoch_start
        val_metrics = evaluate(model, val_loader, device, num_classes, args, epoch)
        row = {
            "epoch": epoch,
            "train_loss": train_loss_sum / max(1, train_count),
            "train_acc": train_acc_sum / max(1, train_count),
            "epoch_seconds": epoch_seconds,
            "train_images_per_sec": train_count / max(epoch_seconds, 1e-9),
            **val_metrics,
        }
        history.append(row)
        print(
            f"{model_name} epoch {epoch:03d}/{args.epochs:03d} "
            f"train_loss={row['train_loss']:.4f} "
            f"train_acc={row['train_acc']:.4f} "
            f"val_loss={row['val_loss']:.4f} "
            f"val_acc={row['val_acc']:.4f}",
            flush=True,
        )

        if val_metrics["val_acc"] > best_val_acc:
            best_val_acc = val_metrics["val_acc"]
            best_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_name": model_name,
                    "model_index": model_index,
                    "classes": args.classes,
                    "state_dict": state_dict_for_save(model),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                best_path,
            )
            print(f"{model_name} saved new best checkpoint: val_acc={best_val_acc:.4f}", flush=True)

    history_path = Path(args.output_dir) / f"{model_index:02d}_{model_name}_history.json"
    save_json(history_path, {"model": model_name, "history": history})
    return {"model": model_name, "best_val_acc": best_val_acc, "checkpoint": str(best_path)}


def train_model_group(
    group: list[tuple[int, str]],
    args: argparse.Namespace,
    train_loaders: DataLoader | dict[int, DataLoader],
    val_loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> list[dict]:
    models = []
    optimizers = []
    histories: dict[int, list[dict]] = {}
    best_val_acc: dict[int, float] = {}
    best_paths: dict[int, Path] = {}

    for model_index, model_name in group:
        set_seed(args.seed + model_index)
        model = MODEL_REGISTRY[model_name](num_classes=num_classes).to(device)
        if args.compile:
            model = torch.compile(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        models.append((model_index, model_name, model))
        optimizers.append(optimizer)
        histories[model_index] = []
        best_val_acc[model_index] = -1.0
        best_paths[model_index] = Path(args.output_dir) / f"{model_index:02d}_{model_name}.pt"

    group_name = "+".join(model_name for _, model_name, _ in models)
    amp_enabled, amp_dtype, scaler = resolve_amp(args, device)
    for epoch in range(1, args.epochs + 1):
        for _, _, model in models:
            model.train()
        maybe_sync(device)
        epoch_start = time.perf_counter()
        train_loss_sum = {model_index: 0.0 for model_index, _, _ in models}
        train_acc_sum = {model_index: 0.0 for model_index, _, _ in models}
        train_count = {model_index: 0 for model_index, _, _ in models}

        if isinstance(train_loaders, dict):
            group_loader_values = [train_loaders[model_index] for model_index, _, _ in models]
            train_iter = zip(*group_loader_values)
            train_steps = min(len(loader) for loader in group_loader_values)
        else:
            train_iter = train_loaders
            train_steps = len(train_loaders)

        progress = progress_iter(train_iter, args, total=train_steps, desc=f"{group_name} epoch {epoch}/{args.epochs}", leave=False)
        for batch in progress:
            if isinstance(train_loaders, dict):
                batch_by_model = {model_index: batch[pos] for pos, (model_index, _, _) in enumerate(models)}
            else:
                batch_by_model = {model_index: batch for model_index, _, _ in models}

            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)

            losses = []
            for model_index, _, model in models:
                images, labels = batch_by_model[model_index]
                images, labels = move_batch(images, labels, device)
                batch_size = labels.shape[0]
                with autocast_context(amp_enabled, amp_dtype, device):
                    output = model(images)
                loss = edl_loss(output["alpha"], labels, epoch, num_classes, args.annealing_epochs)
                losses.append(loss)
                train_loss_sum[model_index] += float(loss.item()) * batch_size
                train_acc_sum[model_index] += accuracy_from_prob(output["prob"].detach(), labels) * batch_size
                train_count[model_index] += batch_size

            total_loss = torch.stack(losses).sum()
            if scaler.is_enabled():
                scaler.scale(total_loss).backward()
                for optimizer in optimizers:
                    scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                for optimizer in optimizers:
                    optimizer.step()

            set_progress_postfix(
                progress,
                args,
                loss=sum(train_loss_sum.values()) / max(1, sum(train_count.values())),
                acc=sum(train_acc_sum.values()) / max(1, sum(train_count.values())),
            )

        maybe_sync(device)
        epoch_seconds = time.perf_counter() - epoch_start
        for model_index, model_name, model in models:
            val_metrics = evaluate(model, val_loader, device, num_classes, args, epoch)
            row = {
                "epoch": epoch,
                "train_loss": train_loss_sum[model_index] / max(1, train_count[model_index]),
                "train_acc": train_acc_sum[model_index] / max(1, train_count[model_index]),
                "epoch_seconds": epoch_seconds,
                "train_images_per_sec": train_count[model_index] / max(epoch_seconds, 1e-9),
                "aggregate_images_per_sec": sum(train_count.values()) / max(epoch_seconds, 1e-9),
                **val_metrics,
            }
            histories[model_index].append(row)
            print(
                f"{model_name} epoch {epoch:03d}/{args.epochs:03d} "
                f"train_loss={row['train_loss']:.4f} "
                f"train_acc={row['train_acc']:.4f} "
                f"val_loss={row['val_loss']:.4f} "
                f"val_acc={row['val_acc']:.4f}",
                flush=True,
            )

            if val_metrics["val_acc"] > best_val_acc[model_index]:
                best_val_acc[model_index] = val_metrics["val_acc"]
                best_paths[model_index].parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_name": model_name,
                        "model_index": model_index,
                        "classes": args.classes,
                        "state_dict": state_dict_for_save(model),
                        "epoch": epoch,
                        "val_metrics": val_metrics,
                    },
                    best_paths[model_index],
                )
                print(
                    f"{model_name} saved new best checkpoint: val_acc={best_val_acc[model_index]:.4f}",
                    flush=True,
                )

    results = []
    for model_index, model_name, _ in models:
        history_path = Path(args.output_dir) / f"{model_index:02d}_{model_name}_history.json"
        save_json(history_path, {"model": model_name, "history": histories[model_index]})
        results.append(
            {
                "model": model_name,
                "best_val_acc": best_val_acc[model_index],
                "checkpoint": str(best_paths[model_index]),
            }
        )
    return results


@torch.no_grad()
def evaluate(model, loader, device, num_classes: int, args, epoch: int) -> dict:
    model.eval()
    amp_enabled, amp_dtype, _ = resolve_amp(args, device)
    loss_sum = 0.0
    acc_sum = 0.0
    count = 0
    for images, labels in loader:
        images, labels = move_batch(images, labels, device)
        with autocast_context(amp_enabled, amp_dtype, device):
            output = model(images)
        loss = edl_loss(output["alpha"], labels, epoch, num_classes, args.annealing_epochs)
        batch_size = labels.shape[0]
        loss_sum += float(loss.item()) * batch_size
        acc_sum += accuracy_from_prob(output["prob"], labels) * batch_size
        count += batch_size
    return {"val_loss": loss_sum / max(1, count), "val_acc": acc_sum / max(1, count)}


def main() -> None:
    args = parse_args()
    selected_classes = parse_classes(args.classes)
    spec = Cifar5Spec(selected_classes)
    model_names = tuple(x.strip() for x in args.models.split(",") if x.strip())
    unknown = [name for name in model_names if name not in MODEL_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown model names: {unknown}; valid={sorted(MODEL_REGISTRY)}")
    if len(model_names) != 5:
        raise ValueError("This first-stage script expects exactly K=5 classifiers.")

    set_seed(args.seed)
    device = torch.device(args.device)
    configure_torch(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    full_train_aug = build_cifar5_dataset(
        args.data_dir,
        train=True,
        classes=selected_classes,
        download=not args.no_download,
        subset_size=args.subset_size,
        augment=True,
        source=args.data_source,
    )
    full_train_eval = build_cifar5_dataset(
        args.data_dir,
        train=True,
        classes=selected_classes,
        download=not args.no_download,
        subset_size=args.subset_size,
        augment=False,
        source=args.data_source,
    )
    if args.per_model_splits:
        train_indices, val_indices, fusion_val_indices, base_split_manifest = stratified_three_way_indices(
            full_train_aug,
            model_train_per_class=args.model_train_per_class,
            checkpoint_val_per_class=args.checkpoint_val_per_class,
            fusion_val_per_class=args.fusion_val_per_class,
            seed=args.seed,
        )
        split_strategy = "stratified-three-way"
    else:
        val_size = max(1, int(len(full_train_aug) * args.val_ratio))
        train_size = len(full_train_aug) - val_size
        perm = torch.randperm(len(full_train_aug), generator=torch.Generator().manual_seed(args.seed)).tolist()
        train_indices = perm[:train_size]
        val_indices = perm[train_size:]
        fusion_val_indices = []
        base_split_manifest = None
        split_strategy = "random"
    train_size = len(train_indices)
    val_size = len(val_indices)
    fusion_val_size = len(fusion_val_indices)
    train_set = Subset(full_train_aug, train_indices)
    val_set = Subset(full_train_eval, val_indices)
    train_loader: DataLoader | None = None
    train_loaders_by_model: dict[int, DataLoader] | None = None
    split_manifest = None
    if args.per_model_splits:
        if args.per_model_train_size > train_size:
            raise ValueError("--per-model-train-size cannot exceed the train pool size")
        model_splits, model_split_manifest = build_per_model_splits(
            full_train_aug,
            train_indices,
            num_models=len(model_names),
            per_model_train_size=args.per_model_train_size,
            overlap_ratio=args.overlap_ratio,
            partition_mode=args.partition_mode,
            seed=args.seed,
        )
        split_manifest = dict(base_split_manifest or {})
        split_manifest.update(
            {
                "num_models": model_split_manifest["num_models"],
                "per_model_train_size": model_split_manifest["per_model_train_size"],
                "overlap_ratio": model_split_manifest["overlap_ratio"],
                "partition_mode": model_split_manifest["partition_mode"],
                "seed": model_split_manifest["seed"],
                "class_ids": model_split_manifest["class_ids"],
                "shared_total": model_split_manifest["shared_total"],
                "private_total": model_split_manifest["private_total"],
                "mean_pairwise_overlap": model_split_manifest["mean_pairwise_overlap"],
                "model_stats": model_split_manifest["model_stats"],
                "model_indices": model_split_manifest["model_indices"],
            }
        )
        split_path = Path(args.output_dir) / "splits.json"
        save_split_manifest(split_path, split_manifest)
        train_loaders_by_model = {
            model_index: make_train_loader(args, Subset(full_train_aug, model_splits[model_index]))
            for model_index in range(len(model_names))
        }
        print(f"wrote split manifest: {split_path}", flush=True)
    else:
        train_loader = make_train_loader(args, train_set)
    val_loader = make_val_loader(args, val_set)

    manifest = {
        "dataset": "CIFAR-5 from CIFAR-10",
        "classes": selected_classes,
        "class_names": spec.names,
        "models": model_names,
        "subset_size": args.subset_size,
        "data_source": args.data_source,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "parallel_models": args.parallel_models,
        "amp": args.amp,
        "compile": args.compile,
        "disable_progress": args.disable_progress,
        "prefetch_factor": args.prefetch_factor,
        "persistent_workers": args.num_workers > 0 and not args.no_persistent_workers,
        "drop_last": not args.no_drop_last,
        "per_model_splits": args.per_model_splits,
        "train_val_split_strategy": split_strategy,
        "overlap_ratio": args.overlap_ratio if args.per_model_splits else None,
        "partition_mode": args.partition_mode if args.per_model_splits else None,
        "per_model_train_size": args.per_model_train_size if args.per_model_splits else None,
        "model_train_per_class": args.model_train_per_class if args.per_model_splits else None,
        "checkpoint_val_per_class": args.checkpoint_val_per_class if args.per_model_splits else None,
        "fusion_val_per_class": args.fusion_val_per_class if args.per_model_splits else None,
        "mean_pairwise_overlap": split_manifest.get("mean_pairwise_overlap") if split_manifest else None,
        "split_manifest": "splits.json" if args.per_model_splits else None,
        "train_size": train_size,
        "checkpoint_val_size": val_size,
        "fusion_val_size": fusion_val_size,
        "device": str(device),
        "results": [],
    }
    print(
        f"Training CIFAR-5: train_size={train_size}, checkpoint_val_size={val_size}, "
        f"fusion_val_size={fusion_val_size}, "
        f"models={','.join(model_names)}, epochs={args.epochs}, "
        f"parallel_models={args.parallel_models}, per_model_splits={args.per_model_splits}, device={device}",
        flush=True,
    )
    indexed_models = list(enumerate(model_names))
    if args.parallel_models <= 1:
        for idx, model_name in indexed_models:
            print(f"start model {idx}: {model_name}", flush=True)
            current_loader = train_loaders_by_model[idx] if train_loaders_by_model is not None else train_loader
            if current_loader is None:
                raise RuntimeError("Internal error: missing train loader")
            manifest["results"].append(
                train_one_model(model_name, idx, args, current_loader, val_loader, device, num_classes=5)
            )
            save_json(Path(args.output_dir) / "manifest.json", manifest)
            print(f"finished model {idx}: {model_name}", flush=True)
    else:
        for group in chunked(indexed_models, args.parallel_models):
            group_name = "+".join(model_name for _, model_name in group)
            print(f"start model group: {group_name}", flush=True)
            current_loaders: DataLoader | dict[int, DataLoader]
            if train_loaders_by_model is not None:
                current_loaders = train_loaders_by_model
            elif train_loader is not None:
                current_loaders = train_loader
            else:
                raise RuntimeError("Internal error: missing train loader")
            manifest["results"].extend(
                train_model_group(group, args, current_loaders, val_loader, device, num_classes=5)
            )
            save_json(Path(args.output_dir) / "manifest.json", manifest)
            print(f"finished model group: {group_name}", flush=True)
    if device.type == "cuda":
        manifest["peak_vram_mb"] = torch.cuda.max_memory_allocated(device) / (1024**2)
        save_json(Path(args.output_dir) / "manifest.json", manifest)
    print(f"wrote manifest: {Path(args.output_dir) / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
