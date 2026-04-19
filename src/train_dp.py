from __future__ import annotations

import random
from typing import Dict

import numpy as np
import torch
from opacus import PrivacyEngine
from torch import nn

from src.data import apply_square_trigger, make_loader
from src.model import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_accuracy(model: nn.Module, x_test: torch.Tensor, y_test: torch.Tensor, device: torch.device) -> float:
    model.eval()
    loader = make_loader(x_test, y_test, batch_size=256, shuffle=False, seed=0)
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
    return correct / max(total, 1)


def evaluate_trigger_success(
    model: nn.Module,
    x_test: torch.Tensor,
    target_label: int,
    trigger_size: int,
    device: torch.device,
) -> float:
    model.eval()
    x_trigger = apply_square_trigger(x_test, size=trigger_size)
    y_target = torch.full((x_trigger.shape[0],), target_label, dtype=torch.long)
    loader = make_loader(x_trigger, y_target, batch_size=256, shuffle=False, seed=0)
    hits = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb).argmax(dim=1)
            hits += (pred == yb).sum().item()
            total += yb.numel()
    return hits / max(total, 1)


def train_dp_once(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    cfg: Dict,
    seed: int,
) -> Dict:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg["model_name"]).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg.get("weight_decay", 0.0))
    criterion = nn.CrossEntropyLoss()

    train_loader = make_loader(
        x_train,
        y_train,
        batch_size=cfg["batch_size"],
        shuffle=True,
        seed=seed,
    )

    privacy_engine = PrivacyEngine(accountant="rdp")
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=cfg["noise_multiplier"],
        max_grad_norm=cfg["max_grad_norm"],
    )

    model.train()
    for _ in range(cfg["epochs"]):
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    epsilon = privacy_engine.get_epsilon(delta=float(cfg["delta"]))
    clean_acc = evaluate_accuracy(model, x_test, y_test, device)
    trigger_asr = evaluate_trigger_success(
        model,
        x_test,
        target_label=cfg["target_label"],
        trigger_size=cfg["trigger_size"],
        device=device,
    )

    return {
        "epsilon_theoretical": float(epsilon),
        "clean_accuracy": float(clean_acc),
        "trigger_success": float(trigger_asr),
    }
