from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tarfile

import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms


CIFAR10_CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


@dataclass(frozen=True)
class Cifar5Spec:
    cifar10_classes: tuple[int, ...] = (0, 1, 2, 3, 4)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(CIFAR10_CLASS_NAMES[i] for i in self.cifar10_classes)

    @property
    def remap(self) -> dict[int, int]:
        return {old: new for new, old in enumerate(self.cifar10_classes)}


class RemappedSubset(Dataset):
    def __init__(self, dataset: Dataset, indices: list[int], remap: dict[int, int]) -> None:
        self.dataset = dataset
        self.indices = indices
        self.remap = remap

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        image, label = self.dataset[self.indices[idx]]
        return image, self.remap[int(label)]


def parse_classes(value: str) -> tuple[int, ...]:
    classes = tuple(int(x.strip()) for x in value.split(",") if x.strip() != "")
    if len(classes) != 5:
        raise ValueError("CIFAR-5 requires exactly five CIFAR-10 class ids")
    if len(set(classes)) != 5:
        raise ValueError("CIFAR-5 class ids must be unique")
    for class_id in classes:
        if not 0 <= class_id <= 9:
            raise ValueError("CIFAR-10 class ids must be in [0,9]")
    return classes


def build_transforms(train: bool):
    if train:
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )


def _safe_extract_tar(archive: Path, target_dir: Path) -> None:
    """Extract a tar archive without allowing path traversal."""
    target_root = target_dir.resolve()
    with tarfile.open(archive, "r:*") as tar:
        for member in tar.getmembers():
            destination = (target_dir / member.name).resolve()
            if not str(destination).startswith(str(target_root)):
                raise RuntimeError(f"Unsafe tar member path: {member.name}")
        tar.extractall(target_dir)


def _ensure_fastai_cifar10(data_dir: str) -> Path | None:
    root = Path(data_dir)
    extracted = root / "cifar10"
    if (extracted / "train").is_dir() and (extracted / "test").is_dir():
        return extracted
    archive = root / "cifar10.tgz"
    if archive.is_file():
        _safe_extract_tar(archive, root)
        if (extracted / "train").is_dir() and (extracted / "test").is_dir():
            return extracted
    return None


def _build_fastai_cifar5_dataset(
    data_dir: str,
    *,
    train: bool,
    classes: tuple[int, ...],
    subset_size: int | None,
    augment: bool,
) -> RemappedSubset:
    cifar10_root = _ensure_fastai_cifar10(data_dir)
    if cifar10_root is None:
        raise FileNotFoundError("fast.ai CIFAR-10 folder/archive was not found")

    split_root = cifar10_root / ("train" if train else "test")
    dataset = datasets.ImageFolder(root=str(split_root), transform=build_transforms(augment))
    selected_names = tuple(CIFAR10_CLASS_NAMES[i] for i in classes)
    class_name_to_selected = {name: idx for idx, name in enumerate(selected_names)}
    remap: dict[int, int] = {}
    for class_name, folder_idx in dataset.class_to_idx.items():
        if class_name in class_name_to_selected:
            remap[folder_idx] = class_name_to_selected[class_name]
    missing = [name for name in selected_names if name not in dataset.class_to_idx]
    if missing:
        raise ValueError(f"Missing selected CIFAR classes in ImageFolder data: {missing}")
    indices = [i for i, (_, label) in enumerate(dataset.samples) if int(label) in remap]
    if subset_size is not None:
        indices = indices[: int(subset_size)]
    return RemappedSubset(dataset, indices, remap)


def build_cifar5_dataset(
    data_dir: str,
    *,
    train: bool,
    classes: tuple[int, ...] = (0, 1, 2, 3, 4),
    download: bool = True,
    subset_size: int | None = None,
    augment: bool | None = None,
    source: str = "auto",
) -> RemappedSubset | Subset:
    spec = Cifar5Spec(classes)
    if augment is None:
        augment = train
    if source not in {"auto", "torchvision", "imagefolder"}:
        raise ValueError("source must be one of: auto, torchvision, imagefolder")
    if source in {"auto", "imagefolder"}:
        try:
            return _build_fastai_cifar5_dataset(
                data_dir,
                train=train,
                classes=classes,
                subset_size=subset_size,
                augment=augment,
            )
        except FileNotFoundError:
            if source == "imagefolder":
                raise

    dataset = datasets.CIFAR10(
        root=data_dir,
        train=train,
        download=download,
        transform=build_transforms(augment),
    )
    indices = [i for i, label in enumerate(dataset.targets) if int(label) in spec.remap]
    if subset_size is not None:
        indices = indices[: int(subset_size)]
    return RemappedSubset(dataset, indices, spec.remap)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    import random
    import numpy as np

    np.random.seed(worker_seed)
    random.seed(worker_seed)
