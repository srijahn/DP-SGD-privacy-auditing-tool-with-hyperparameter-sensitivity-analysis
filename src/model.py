from __future__ import annotations

import torch
from torch import nn


class LogisticRegressionNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SmallMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(model_name: str) -> nn.Module:
    if model_name == "logreg":
        return LogisticRegressionNet()
    if model_name == "mlp":
        return SmallMLP()
    raise ValueError(f"Unsupported model_name={model_name}")
