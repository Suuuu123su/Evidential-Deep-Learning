from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset


@dataclass(frozen=True)
class SplitConfig:
    num_models: int
    per_model_train_size: int
    overlap_ratio: float
    partition_mode: str
    seed: int


def dataset_label(dataset: Dataset, idx: int) -> int:
    """Return the remapped class label for a dataset index without loading pixels when possible."""
    if hasattr(dataset, "indices") and hasattr(dataset, "remap") and hasattr(dataset, "dataset"):
        raw_idx = int(dataset.indices[idx])
        base = dataset.dataset
        if hasattr(base, "targets"):
            raw_label = int(base.targets[raw_idx])
        elif hasattr(base, "samples"):
            raw_label = int(base.samples[raw_idx][1])
        else:
            _, raw_label = base[raw_idx]
            raw_label = int(raw_label)
        return int(dataset.remap[raw_label])

    if hasattr(dataset, "targets"):
        return int(dataset.targets[idx])
    if hasattr(dataset, "samples"):
        return int(dataset.samples[idx][1])
    _, label = dataset[idx]
    return int(label)


def indices_by_class(dataset: Dataset, candidate_indices: list[int]) -> dict[int, list[int]]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for idx in candidate_indices:
        grouped[dataset_label(dataset, idx)].append(int(idx))
    return dict(grouped)


def stratified_train_val_indices(
    dataset: Dataset,
    *,
    val_ratio: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be in (0, 1) for stratified splitting")
    rng = random.Random(seed)
    all_indices = list(range(len(dataset)))
    grouped = indices_by_class(dataset, all_indices)
    train_indices: list[int] = []
    val_indices: list[int] = []
    for class_id in sorted(grouped):
        indices = grouped[class_id]
        rng.shuffle(indices)
        val_size = max(1, int(round(len(indices) * val_ratio)))
        val_indices.extend(indices[:val_size])
        train_indices.extend(indices[val_size:])
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def stratified_three_way_indices(
    dataset: Dataset,
    *,
    model_train_per_class: int,
    checkpoint_val_per_class: int,
    fusion_val_per_class: int,
    seed: int,
) -> tuple[list[int], list[int], list[int], dict[str, Any]]:
    rng = random.Random(seed)
    grouped = indices_by_class(dataset, list(range(len(dataset))))
    model_train_indices: list[int] = []
    checkpoint_val_indices: list[int] = []
    fusion_val_indices: list[int] = []
    split_stats: list[dict[str, Any]] = []

    for class_id in sorted(grouped):
        indices = grouped[class_id]
        rng.shuffle(indices)
        required = model_train_per_class + checkpoint_val_per_class + fusion_val_per_class
        if required > len(indices):
            raise ValueError(
                f"Not enough samples for class {class_id}: "
                f"need {required}, have {len(indices)}"
            )
        model_part = indices[:model_train_per_class]
        checkpoint_part = indices[model_train_per_class : model_train_per_class + checkpoint_val_per_class]
        fusion_part = indices[
            model_train_per_class + checkpoint_val_per_class :
            model_train_per_class + checkpoint_val_per_class + fusion_val_per_class
        ]
        model_train_indices.extend(model_part)
        checkpoint_val_indices.extend(checkpoint_part)
        fusion_val_indices.extend(fusion_part)
        split_stats.append(
            {
                "class_id": class_id,
                "model_train_count": len(model_part),
                "checkpoint_val_count": len(checkpoint_part),
                "fusion_val_count": len(fusion_part),
            }
        )

    rng.shuffle(model_train_indices)
    rng.shuffle(checkpoint_val_indices)
    rng.shuffle(fusion_val_indices)
    manifest = {
        "split_strategy": "stratified-three-way",
        "model_train_per_class": model_train_per_class,
        "checkpoint_val_per_class": checkpoint_val_per_class,
        "fusion_val_per_class": fusion_val_per_class,
        "model_train_size": len(model_train_indices),
        "checkpoint_val_size": len(checkpoint_val_indices),
        "fusion_val_size": len(fusion_val_indices),
        "model_train_indices": model_train_indices,
        "checkpoint_val_indices": checkpoint_val_indices,
        "fusion_val_indices": fusion_val_indices,
        "model_train_hash": _hash_indices(model_train_indices),
        "checkpoint_val_hash": _hash_indices(checkpoint_val_indices),
        "fusion_val_hash": _hash_indices(fusion_val_indices),
        "class_stats": split_stats,
    }
    return model_train_indices, checkpoint_val_indices, fusion_val_indices, manifest


def _balanced_counts(total: int, class_count: int) -> list[int]:
    base = total // class_count
    remainder = total % class_count
    return [base + (1 if i < remainder else 0) for i in range(class_count)]


def _skewed_counts(total: int, class_count: int, target_class: int, target_fraction: float = 0.5) -> list[int]:
    if total == 0:
        return [0] * class_count
    target = min(total, int(round(total * target_fraction)))
    counts = [0] * class_count
    counts[target_class] = target
    remaining = total - target
    other_classes = [i for i in range(class_count) if i != target_class]
    for offset, class_id in enumerate(other_classes):
        counts[class_id] = remaining // len(other_classes) + (1 if offset < remaining % len(other_classes) else 0)
    return counts


def _rebalance_counts_to_capacity(
    counts_by_model: list[list[int]],
    capacity_by_class: list[int],
    target_class_by_model: list[int],
) -> None:
    required_by_class = [sum(counts[class_pos] for counts in counts_by_model) for class_pos in range(len(capacity_by_class))]
    if sum(required_by_class) > sum(capacity_by_class):
        raise ValueError("Private split requires more unique samples than the training pool provides")

    while True:
        overfull = [i for i, required in enumerate(required_by_class) if required > capacity_by_class[i]]
        if not overfull:
            return
        underfull = [i for i, required in enumerate(required_by_class) if required < capacity_by_class[i]]
        if not underfull:
            raise ValueError("Cannot rebalance private counts: no class has spare capacity")

        over_class = max(overfull, key=lambda i: required_by_class[i] - capacity_by_class[i])
        donor_models = sorted(
            range(len(counts_by_model)),
            key=lambda model_index: (
                target_class_by_model[model_index] == over_class,
                -counts_by_model[model_index][over_class],
                model_index,
            ),
        )
        receive_classes = sorted(
            underfull,
            key=lambda class_pos: (-(capacity_by_class[class_pos] - required_by_class[class_pos]), class_pos),
        )

        moved = False
        for model_index in donor_models:
            if counts_by_model[model_index][over_class] <= 0:
                continue
            for receive_class in receive_classes:
                if receive_class == over_class:
                    continue
                counts_by_model[model_index][over_class] -= 1
                counts_by_model[model_index][receive_class] += 1
                required_by_class[over_class] -= 1
                required_by_class[receive_class] += 1
                moved = True
                break
            if moved:
                break
        if not moved:
            raise ValueError("Cannot rebalance private counts without making a class count negative")


def _hash_indices(indices: list[int]) -> str:
    digest = hashlib.sha256()
    digest.update(",".join(str(i) for i in sorted(indices)).encode("utf-8"))
    return digest.hexdigest()[:16]


def build_per_model_splits(
    dataset: Dataset,
    train_indices: list[int],
    *,
    num_models: int,
    per_model_train_size: int,
    overlap_ratio: float,
    partition_mode: str,
    seed: int,
) -> tuple[list[list[int]], dict[str, Any]]:
    if not 0.0 <= overlap_ratio <= 1.0:
        raise ValueError("overlap_ratio must be in [0, 1]")
    if partition_mode not in {"stratified-balanced", "class-skewed-private"}:
        raise ValueError("partition_mode must be stratified-balanced or class-skewed-private")
    if per_model_train_size <= 0:
        raise ValueError("per_model_train_size must be positive")
    if num_models <= 0:
        raise ValueError("num_models must be positive")

    rng = random.Random(seed)
    grouped = indices_by_class(dataset, train_indices)
    class_ids = sorted(grouped)
    class_count = len(class_ids)
    if class_count == 0:
        raise ValueError("No classes found in train_indices")
    if per_model_train_size > len(train_indices):
        raise ValueError("per_model_train_size cannot exceed available train pool")

    for values in grouped.values():
        rng.shuffle(values)

    per_model_counts = _balanced_counts(per_model_train_size, class_count)
    shared_total = int(round(per_model_train_size * overlap_ratio))
    shared_counts = _balanced_counts(shared_total, class_count)
    private_total = per_model_train_size - shared_total

    shared_by_class: dict[int, list[int]] = {}
    remaining_by_class: dict[int, list[int]] = {}
    for class_pos, class_id in enumerate(class_ids):
        available = grouped[class_id]
        need = shared_counts[class_pos]
        if need > len(available):
            raise ValueError(f"Not enough samples for shared class {class_id}: need {need}, have {len(available)}")
        shared_by_class[class_id] = available[:need]
        remaining_by_class[class_id] = available[need:]

    private_counts_by_model: list[list[int]] = []
    target_class_by_model: list[int] = []
    for model_index in range(num_models):
        if partition_mode == "stratified-balanced":
            private_counts = [per_model_counts[i] - shared_counts[i] for i in range(class_count)]
            target_class_by_model.append(-1)
        else:
            target_class_pos = model_index % class_count
            private_counts = _skewed_counts(private_total, class_count, target_class_pos)
            target_class_by_model.append(target_class_pos)
        if sum(private_counts) != private_total:
            raise RuntimeError("Internal split error: private counts do not sum to private_total")
        private_counts_by_model.append(private_counts)

    if partition_mode == "class-skewed-private":
        _rebalance_counts_to_capacity(
            private_counts_by_model,
            [len(remaining_by_class[class_id]) for class_id in class_ids],
            target_class_by_model,
        )

    required_by_class = [0] * class_count
    for private_counts in private_counts_by_model:
        for class_pos, count in enumerate(private_counts):
            required_by_class[class_pos] += count
    for class_pos, class_id in enumerate(class_ids):
        available = len(remaining_by_class[class_id])
        if required_by_class[class_pos] > available:
            raise ValueError(
                f"Not enough private samples for class {class_id}: "
                f"need {required_by_class[class_pos]}, have {available}"
            )

    cursors = {class_id: 0 for class_id in class_ids}
    model_splits: list[list[int]] = []
    for model_index in range(num_models):
        split: list[int] = []
        for class_id in class_ids:
            split.extend(shared_by_class[class_id])
        for class_pos, class_id in enumerate(class_ids):
            count = private_counts_by_model[model_index][class_pos]
            start = cursors[class_id]
            end = start + count
            split.extend(remaining_by_class[class_id][start:end])
            cursors[class_id] = end
        rng.shuffle(split)
        model_splits.append(split)

    shared_set = set()
    for values in shared_by_class.values():
        shared_set.update(values)
    stats = []
    for model_index, split in enumerate(model_splits):
        label_counts = {str(class_id): 0 for class_id in class_ids}
        for idx in split:
            label_counts[str(dataset_label(dataset, idx))] += 1
        stats.append(
            {
                "model_index": model_index,
                "size": len(split),
                "shared_count": sum(1 for idx in split if idx in shared_set),
                "private_count": sum(1 for idx in split if idx not in shared_set),
                "class_counts": label_counts,
                "indices_hash": _hash_indices(split),
            }
        )

    actual_overlaps = []
    for i in range(num_models):
        for j in range(i + 1, num_models):
            a = set(model_splits[i])
            b = set(model_splits[j])
            actual_overlaps.append(len(a & b) / max(1, per_model_train_size))

    manifest = {
        "num_models": num_models,
        "per_model_train_size": per_model_train_size,
        "overlap_ratio": overlap_ratio,
        "partition_mode": partition_mode,
        "seed": seed,
        "class_ids": class_ids,
        "shared_total": shared_total,
        "private_total": private_total,
        "mean_pairwise_overlap": sum(actual_overlaps) / max(1, len(actual_overlaps)),
        "model_stats": stats,
        "model_indices": model_splits,
    }
    return model_splits, manifest


def save_split_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
