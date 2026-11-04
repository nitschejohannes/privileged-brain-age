#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from teacher_and_student_models import (
    GDModelConfig,
    StudentClassifier,
    logits_to_expected_age,
    soft_cross_entropy_from_logits,
)

@torch.no_grad()
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return mae, rmse, r2

def make_baseline_model(
    x_dim: int,
    n_classes: int,
    cfg: Optional[GDModelConfig] = None,
) -> StudentClassifier:
    if cfg is None:
        cfg = GDModelConfig()
    return StudentClassifier(x_dim=x_dim, n_classes=n_classes, cfg=cfg)

def train_baseline(
    model: StudentClassifier,
    train_loader,
    val_loader,
    *,
    age_grid: torch.Tensor,
    device: torch.device,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 200,
    patience: int = 20,
    verbose: bool = True,
) -> Dict[str, list]:
    """
    Train baseline with supervised soft cross-entropy to Gaussian-smoothed age targets.
    Expects batches: (x, y, y_prob)
    """
    model.to(device)
    age_grid = age_grid.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = float("inf")
    best_state = None
    bad = 0

    hist = {
        "train_loss": [],
        "train_mae": [],
        "train_rmse": [],
        "train_r2": [],
        "val_loss": [],
        "val_mae": [],
        "val_rmse": [],
        "val_r2": [],
    }

    def run_epoch(loader, train: bool):
        model.train(train)
        total_loss = 0.0
        n = 0
        y_all, p_all = [], []

        for x, y, y_prob in loader:
            x = x.to(device)
            y = y.to(device).float().view(-1)
            y_prob = y_prob.to(device).float()

            if train:
                opt.zero_grad(set_to_none=True)

            logits = model(x)
            loss = soft_cross_entropy_from_logits(logits, y_prob, reduction="mean")

            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()

            pred_age = logits_to_expected_age(logits, age_grid, temperature=1.0)

            bs = y.shape[0]
            total_loss += float(loss.detach().cpu()) * bs
            n += bs
            y_all.append(y.detach().cpu().numpy().reshape(-1))
            p_all.append(pred_age.detach().cpu().numpy().reshape(-1))

        y_cat = np.concatenate(y_all, axis=0)
        p_cat = np.concatenate(p_all, axis=0)
        mae, rmse, r2 = compute_metrics(y_cat, p_cat)
        return total_loss / max(n, 1), mae, rmse, r2

    for epoch in range(1, max_epochs + 1):
        tr_loss, tr_mae, tr_rmse, tr_r2 = run_epoch(train_loader, train=True)
        va_loss, va_mae, va_rmse, va_r2 = run_epoch(val_loader, train=False)

        hist["train_loss"].append(tr_loss)
        hist["train_mae"].append(tr_mae)
        hist["train_rmse"].append(tr_rmse)
        hist["train_r2"].append(tr_r2)
        hist["val_loss"].append(va_loss)
        hist["val_mae"].append(va_mae)
        hist["val_rmse"].append(va_rmse)
        hist["val_r2"].append(va_r2)

        if verbose:
            print(
                f"[BL] epoch {epoch:03d} | "
                f"train ce {tr_loss:.4f} mae {tr_mae:.3f} rmse {tr_rmse:.3f} r2 {tr_r2:.3f} | "
                f"val ce {va_loss:.4f} mae {va_mae:.3f} rmse {va_rmse:.3f} r2 {va_r2:.3f}"
            )

        if va_loss < best_val - 1e-6:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"[BL] early stopping at epoch {epoch} (best val ce {best_val:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return hist