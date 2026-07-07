from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(labels, num_classes=num_classes).float()


def dirichlet_kl_to_uniform(alpha: torch.Tensor) -> torch.Tensor:
    """KL[Dir(alpha) || Dir(1)] per sample."""
    num_classes = alpha.shape[-1]
    beta = torch.ones_like(alpha)
    sum_alpha = alpha.sum(dim=-1, keepdim=True)
    sum_beta = beta.sum(dim=-1, keepdim=True)
    ln_b = torch.lgamma(sum_alpha) - torch.lgamma(alpha).sum(dim=-1, keepdim=True)
    ln_b_uni = torch.lgamma(beta).sum(dim=-1, keepdim=True) - torch.lgamma(sum_beta)
    digamma_term = ((alpha - beta) * (torch.digamma(alpha) - torch.digamma(sum_alpha))).sum(
        dim=-1, keepdim=True
    )
    return (ln_b + ln_b_uni + digamma_term).squeeze(-1)


def edl_loss(alpha: torch.Tensor, labels: torch.Tensor, epoch: int, num_classes: int, annealing_epochs: int) -> torch.Tensor:
    alpha = alpha.float()
    y = one_hot(labels, num_classes)
    strength = alpha.sum(dim=-1, keepdim=True)
    data_fit = torch.sum(y * (torch.log(strength) - torch.log(alpha)), dim=-1)
    alpha_tilde = y + (1.0 - y) * alpha
    anneal = min(1.0, epoch / max(1, annealing_epochs))
    kl = dirichlet_kl_to_uniform(alpha_tilde)
    return torch.mean(data_fit + anneal * kl)


@torch.no_grad()
def accuracy_from_prob(prob: torch.Tensor, labels: torch.Tensor) -> float:
    pred = prob.argmax(dim=-1)
    return float((pred == labels).float().mean().item())


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
