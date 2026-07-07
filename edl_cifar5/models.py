from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def evidence_from_logits(logits: torch.Tensor, kind: str = "softplus") -> torch.Tensor:
    if kind == "relu":
        return F.relu(logits)
    if kind == "softplus":
        return F.softplus(logits)
    raise ValueError(f"Unknown evidence activation: {kind}")


class EDLClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, feature_dim: int, num_classes: int = 5, evidence: str = "softplus") -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(feature_dim, num_classes)
        self.evidence = evidence

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(x)
        logits = self.head(features)
        evidence = evidence_from_logits(logits, self.evidence)
        alpha = evidence + 1.0
        prob = alpha / alpha.sum(dim=-1, keepdim=True)
        return {"logits": logits, "evidence": evidence, "alpha": alpha, "prob": prob}


class FlattenBackbone(nn.Module):
    def __init__(self, modules: nn.Module, feature_dim: int) -> None:
        super().__init__()
        self.modules_ = modules
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.modules_(x)
        return torch.flatten(x, 1)


def lenet_cifar(num_classes: int = 5) -> EDLClassifier:
    backbone = nn.Sequential(
        nn.Conv2d(3, 6, kernel_size=5),
        nn.Tanh(),
        nn.AvgPool2d(2),
        nn.Conv2d(6, 16, kernel_size=5),
        nn.Tanh(),
        nn.AvgPool2d(2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120),
        nn.Tanh(),
        nn.Linear(120, 84),
        nn.Tanh(),
    )
    return EDLClassifier(backbone, 84, num_classes=num_classes)


def small_cnn(num_classes: int = 5) -> EDLClassifier:
    backbone = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    )
    return EDLClassifier(backbone, 128, num_classes=num_classes)


def small_vgg(num_classes: int = 5) -> EDLClassifier:
    backbone = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 32, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    )
    return EDLClassifier(backbone, 128, num_classes=num_classes)


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.skip(x)
        return F.relu(out, inplace=True)


def tiny_resnet(num_classes: int = 5) -> EDLClassifier:
    backbone = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1, bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        BasicBlock(32, 32),
        BasicBlock(32, 64, stride=2),
        BasicBlock(64, 128, stride=2),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    )
    return EDLClassifier(backbone, 128, num_classes=num_classes)


class TinyViTBackbone(nn.Module):
    def __init__(self, image_size: int = 32, patch_size: int = 4, embed_dim: int = 96, depth: int = 2, heads: int = 4) -> None:
        super().__init__()
        patch_count = (image_size // patch_size) ** 2
        self.patch = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos = nn.Parameter(torch.zeros(1, patch_count + 1, embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=embed_dim * 2,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch(x)
        x = x.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos[:, : x.shape[1]]
        x = self.encoder(x)
        return self.norm(x[:, 0])


def tiny_vit(num_classes: int = 5) -> EDLClassifier:
    return EDLClassifier(TinyViTBackbone(), 96, num_classes=num_classes)


MODEL_REGISTRY = {
    "lenet": lenet_cifar,
    "small_cnn": small_cnn,
    "small_vgg": small_vgg,
    "tiny_resnet": tiny_resnet,
    "tiny_vit": tiny_vit,
}

