#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from teacher_and_student_models import (
    GDModelConfig,
    StudentClassifier,
    TeacherClassifier,
    combined_student_classification_loss,
    gaussian_soft_labels,
    logits_to_expected_age,
    logits_to_probs,
    make_age_grid,
    soft_cross_entropy_from_logits,
)
from baseline import make_baseline_model
from pipeline_deconfound import (
    FoldDeconfounder,
    resolve_confound_cols,
    required_confounds_for,
    split_features_and_qc,
)

# Identifies which KD method produced the predictions (written into output CSVs).
KD_METHOD_TAG = "rkd_distance_residual_bag"

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def parse_cols(s: str) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    return [c.strip() for c in s.split(",") if c.strip()]

def parse_int_tuple(s: str) -> Tuple[int, ...]:
    s = (s or "").strip()
    if not s:
        return tuple()
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())

def cols_by_prefix(df: pd.DataFrame, prefixes: List[str]) -> List[str]:
    out: List[str] = []
    for p in prefixes:
        if not p:
            continue
        out.extend([c for c in df.columns if c.startswith(p)])
    seen = set()
    uniq_cols = []
    for c in out:
        if c not in seen:
            uniq_cols.append(c)
            seen.add(c)
    return uniq_cols

def uniq(seq: Sequence[str]) -> List[str]:
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

def assert_unique_eids(df_obj: pd.DataFrame, *, name: str, eid_col: str) -> None:
    if eid_col not in df_obj.columns:
        raise ValueError(f"{name} CSV missing eid_col={eid_col}")
    dup_mask = df_obj[eid_col].duplicated(keep=False)
    if dup_mask.any():
        dup_vals = df_obj.loc[dup_mask, eid_col]
        examples = dup_vals.astype(str).drop_duplicates().head(10).tolist()
        raise ValueError(
            f"{name} contains duplicated {eid_col} values. Example IDs: {examples}"
        )

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))
    if len(y_true) < 2 or np.nanstd(y_true) == 0:
        r2 = np.nan
    else:
        r2 = float(r2_score(y_true, y_pred))
    if len(y_true) < 2 or np.nanstd(y_true) == 0 or np.nanstd(y_pred) == 0:
        r = np.nan
    else:
        r = float(pearsonr(y_true, y_pred)[0])
    return mae, rmse, r2, r

def bag_stats(x: np.ndarray) -> Tuple[float, float]:
    return float(np.mean(x)), float(np.std(x))

def median_int(values: Sequence[int]) -> int:
    vals = [int(v) for v in values if v is not None and int(v) > 0]
    if not vals:
        raise ValueError("No positive epoch counts provided.")
    return int(round(float(np.median(vals))))

def log_sex_distribution(df_subset: pd.DataFrame, sex_col: str, name: str) -> None:
    if sex_col not in df_subset.columns:
        print(f"[sanity] {name}: sex_col='{sex_col}' not present")
        return
    s = df_subset[sex_col].dropna().to_numpy()
    if len(s) == 0:
        print(f"[sanity] {name}: no non-missing sex values")
        return
    vals, counts = np.unique(s, return_counts=True)
    total = counts.sum()
    parts = [f"{v}:{c} ({c/total:.2%})" for v, c in zip(vals, counts)]
    print(f"[sanity] {name} sex distribution | N={total} | " + " | ".join(parts))

# -----------------------------------------------------------------------------
# Stratification
# -----------------------------------------------------------------------------
def make_age_bins(age: np.ndarray, n_bins: int) -> np.ndarray:
    s = pd.Series(age)
    try:
        bins = pd.qcut(s, q=n_bins, labels=False, duplicates="drop")
        return bins.to_numpy().astype(int)
    except Exception:
        bins = pd.cut(s, bins=n_bins, labels=False)
        return bins.to_numpy().astype(int)

def infer_site_label_from_dummies(df: pd.DataFrame, prefix: str) -> Optional[np.ndarray]:
    site_cols = [c for c in df.columns if c.startswith(prefix)]
    if len(site_cols) == 0:
        return None
    M = df[site_cols].to_numpy()
    idx = np.argmax(M, axis=1)
    maxv = np.max(M, axis=1)
    labels = np.array([site_cols[i] for i in idx], dtype=str)
    labels[maxv <= 0] = "site_unknown"
    return labels

def make_strata(
    df: pd.DataFrame,
    *,
    age_col: str,
    n_age_bins: int,
    stratify_site: bool,
    site_prefix: str,
    stratify_sex: bool,
    sex_col: str,
) -> np.ndarray:
    def as_str_array(x: Any) -> np.ndarray:
        return np.asarray(x, dtype=str)

    age_bins = as_str_array(make_age_bins(df[age_col].to_numpy(), n_bins=n_age_bins))
    parts = [age_bins]

    if stratify_site:
        site_labels = infer_site_label_from_dummies(df, prefix=site_prefix)
        if site_labels is None:
            raise ValueError(f"stratify_site=True but no columns start with prefix='{site_prefix}'.")
        parts.append(as_str_array(site_labels))

    if stratify_sex:
        if sex_col not in df.columns:
            raise ValueError(f"stratify_sex=True but sex_col='{sex_col}' not found.")
        parts.append(as_str_array(df[sex_col].to_numpy()))

    strata = parts[0]
    for p in parts[1:]:
        strata = np.char.add(np.char.add(strata, "|"), p)
    return strata

# -----------------------------------------------------------------------------
# BAG bias correction
# -----------------------------------------------------------------------------
def fit_bag_bias_correction(y_true_train: np.ndarray, y_pred_train: np.ndarray) -> Tuple[float, float]:
    age = y_true_train.reshape(-1)
    bag = y_pred_train.reshape(-1) - age
    A = np.vstack([age, np.ones_like(age)]).T
    coef, *_ = np.linalg.lstsq(A, bag, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    return a, b

def apply_bag_bias_correction(y_true: np.ndarray, y_pred: np.ndarray, a: float, b: float) -> Tuple[np.ndarray, np.ndarray]:
    age = y_true.reshape(-1)
    bag = y_pred.reshape(-1) - age
    bag_corr = bag - (a * age + b)
    pred_corr = age + bag_corr
    return pred_corr.astype(np.float32), bag_corr.astype(np.float32)

# -----------------------------------------------------------------------------
# Preprocessing
# -----------------------------------------------------------------------------
def scale_X(train: np.ndarray, val: np.ndarray, test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sc = StandardScaler()
    tr = sc.fit_transform(train).astype(np.float32)
    va = sc.transform(val).astype(np.float32)
    te = sc.transform(test).astype(np.float32)
    return tr, va, te

def fit_standard_scaler(train: np.ndarray) -> StandardScaler:
    sc = StandardScaler()
    sc.fit(train)
    return sc

def transform_with_scaler(x: np.ndarray, sc: StandardScaler) -> np.ndarray:
    return sc.transform(x).astype(np.float32)

def fit_student_feature_preprocessor(
    train_raw: np.ndarray,
    *,
    verbose: bool = False,
    tag: str = "student",
) -> Dict[str, Any]:
    """
    Student/baseline feature preprocessing: z-scoring fitted on the training fold
    only. (The previous Yeo-Johnson power transform has been removed.)
    """
    x_train = np.asarray(train_raw, dtype=np.float64)
    sc = StandardScaler()
    sc.fit(x_train)
    if verbose:
        print(f"[{tag}] fitted StandardScaler on {x_train.shape[1]} regular features")
    return {"scaler": sc}

def transform_student_features(x_raw: np.ndarray, prep: Dict[str, Any]) -> np.ndarray:
    x = np.asarray(x_raw, dtype=np.float64)
    x = prep["scaler"].transform(x)
    return x.astype(np.float32)

# -----------------------------------------------------------------------------
# Soft targets / loaders / gradient diagnostics
# -----------------------------------------------------------------------------
def make_soft_targets_np(y: np.ndarray, age_grid: np.ndarray, sigma: float) -> np.ndarray:
    y_t = torch.from_numpy(np.asarray(y, dtype=np.float32))
    grid_t = torch.from_numpy(np.asarray(age_grid, dtype=np.float32))
    probs = gaussian_soft_labels(y_t, grid_t, sigma=sigma)
    return probs.cpu().numpy().astype(np.float32)

def _safe_drop_last(n_samples: int, batch_size: int) -> bool:
    """
    drop_last=True only when there is more than one batch worth of data.
    Useful for RKD-trained loaders where a tiny final batch gives a degenerate
    relational signal. Eval loaders should always pass drop_last=False.
    """
    return n_samples > batch_size

def make_soft_loader(X: np.ndarray, y: np.ndarray, y_prob: np.ndarray, bs: int, shuffle: bool, drop_last: bool = False) -> DataLoader:
    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y).float().view(-1, 1)
    pt = torch.from_numpy(y_prob).float()
    ds = TensorDataset(Xt, yt, pt)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=drop_last)

def make_eval_loader(X: np.ndarray, y: np.ndarray, bs: int, shuffle: bool = False) -> DataLoader:
    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y).float().view(-1, 1)
    ds = TensorDataset(Xt, yt)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=False)

def make_kd_loader(
    X: np.ndarray,
    y: np.ndarray,
    y_prob: np.ndarray,
    teacher_logits: np.ndarray,
    bs: int,
    shuffle: bool,
    *,
    drop_last: bool = False,
) -> DataLoader:
    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y).float().view(-1, 1)
    pt = torch.from_numpy(y_prob).float()
    lt = torch.from_numpy(teacher_logits).float()
    ds = TensorDataset(Xt, yt, pt, lt)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=drop_last)


def get_student_shared_params(student: torch.nn.Module) -> List[torch.nn.Parameter]:
    return [p for p in student.parameters() if p.requires_grad]


def _flatten_grads(
    grads: List[Optional[torch.Tensor]],
    params: List[torch.nn.Parameter],
) -> torch.Tensor:
    flats: List[torch.Tensor] = []
    for g, p in zip(grads, params):
        if g is None:
            flats.append(torch.zeros_like(p, device=p.device).reshape(-1))
        else:
            flats.append(g.detach().reshape(-1))
    if not flats:
        return torch.zeros(0, dtype=torch.float32)
    return torch.cat(flats, dim=0)

def compute_two_loss_grad_stats(
    loss_sup: torch.Tensor,
    loss_kd: torch.Tensor,
    params: List[torch.nn.Parameter],
    *,
    keep_graph_for_backward: bool,
) -> Tuple[float, float, float]:
    sup_grads = torch.autograd.grad(
        loss_sup,
        params,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    kd_grads = torch.autograd.grad(
        loss_kd,
        params,
        retain_graph=keep_graph_for_backward,
        create_graph=False,
        allow_unused=True,
    )

    sup_vec = _flatten_grads(list(sup_grads), params)
    kd_vec = _flatten_grads(list(kd_grads), params)

    sup_norm = float(torch.linalg.norm(sup_vec).cpu()) if sup_vec.numel() > 0 else 0.0
    kd_norm = float(torch.linalg.norm(kd_vec).cpu()) if kd_vec.numel() > 0 else 0.0

    if sup_vec.numel() == 0 or kd_vec.numel() == 0 or sup_norm == 0.0 or kd_norm == 0.0:
        cos_sim = np.nan
    else:
        cos_sim = float(F.cosine_similarity(sup_vec.unsqueeze(0), kd_vec.unsqueeze(0), dim=1).cpu().item())

    return sup_norm, kd_norm, cos_sim

@dataclass
class TrainResult:
    history: Dict[str, List[float]]
    best_epoch: int

# -----------------------------------------------------------------------------
# Training / prediction
# -----------------------------------------------------------------------------
def train_supervised_classifier(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    *,
    age_grid: torch.Tensor,
    device: torch.device,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    verbose: bool,
    tag: str,
) -> TrainResult:
    model.to(device)
    age_grid = age_grid.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_metric = float("inf")
    best_state = None
    best_epoch = max_epochs
    bad = 0

    hist = {
        "train_loss": [],
        "train_mae": [],
        "train_rmse": [],
        "train_r2": [],
        "train_r": [],
        "val_loss": [],
        "val_mae": [],
        "val_rmse": [],
        "val_r2": [],
        "val_r": [],
    }

    def run_epoch(loader: DataLoader, is_train: bool) -> Tuple[float, float, float, float, float]:
        model.train(is_train)
        total_loss = 0.0
        n = 0
        y_all: List[np.ndarray] = []
        pred_all: List[np.ndarray] = []

        for x, y, y_prob in loader:
            x = x.to(device)
            y = y.to(device).float().view(-1)
            y_prob = y_prob.to(device).float()

            if is_train:
                opt.zero_grad(set_to_none=True)

            logits = model(x)
            loss = soft_cross_entropy_from_logits(logits, y_prob, reduction="mean")

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()

            pred_age = logits_to_expected_age(logits, age_grid, temperature=1.0)

            bs = y.shape[0]
            total_loss += float(loss.detach().cpu()) * bs
            n += bs
            y_all.append(y.detach().cpu().numpy().reshape(-1))
            pred_all.append(pred_age.detach().cpu().numpy().reshape(-1))

        y_cat = np.concatenate(y_all, axis=0)
        p_cat = np.concatenate(pred_all, axis=0)
        mae, rmse, r2, r = compute_metrics(y_cat, p_cat)
        return total_loss / max(n, 1), mae, rmse, r2, r

    for epoch in range(1, max_epochs + 1):
        tr_loss, tr_mae, tr_rmse, tr_r2, tr_r = run_epoch(train_loader, is_train=True)
        hist["train_loss"].append(tr_loss)
        hist["train_mae"].append(tr_mae)
        hist["train_rmse"].append(tr_rmse)
        hist["train_r2"].append(tr_r2)
        hist["train_r"].append(tr_r)

        if val_loader is not None:
            va_loss, va_mae, va_rmse, va_r2, va_r = run_epoch(val_loader, is_train=False)
            hist["val_loss"].append(va_loss)
            hist["val_mae"].append(va_mae)
            hist["val_rmse"].append(va_rmse)
            hist["val_r2"].append(va_r2)
            hist["val_r"].append(va_r)
            current_metric = va_loss
        else:
            va_loss = va_mae = va_rmse = va_r2 = va_r = np.nan
            hist["val_loss"].append(np.nan)
            hist["val_mae"].append(np.nan)
            hist["val_rmse"].append(np.nan)
            hist["val_r2"].append(np.nan)
            hist["val_r"].append(np.nan)
            current_metric = tr_loss

        if verbose:
            print(
                f"[{tag}] epoch {epoch:03d} | "
                f"train ce {tr_loss:.4f} mae {tr_mae:.3f} rmse {tr_rmse:.3f} r2 {tr_r2:.3f} r {tr_r:.3f} | "
                f"val ce {va_loss:.4f} mae {va_mae:.3f} rmse {va_rmse:.3f} r2 {va_r2:.3f} r {va_r:.3f}"
            )

        if val_loader is not None:
            if current_metric < best_metric - 1e-6:
                best_metric = current_metric
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    if verbose:
                        print(f"[{tag}] early stopping at epoch {epoch} (best epoch {best_epoch}, best val loss {best_metric:.4f})")
                    break
        else:
            best_epoch = max_epochs

    if val_loader is not None and best_state is not None:
        model.load_state_dict(best_state)

    return TrainResult(history=hist, best_epoch=best_epoch)

@torch.no_grad()
def predict_teacher(
    teacher: TeacherClassifier,
    loader: DataLoader,
    *,
    age_grid: torch.Tensor,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    teacher.eval()
    age_grid = age_grid.to(device)
    y_all, pred_all, logits_all, probs_all = [], [], [], []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device).float().view(-1)

        logits = teacher(x)
        probs = logits_to_probs(logits, temperature=1.0)
        pred_age = logits_to_expected_age(logits, age_grid, temperature=1.0)

        y_all.append(y.detach().cpu().numpy().reshape(-1))
        pred_all.append(pred_age.detach().cpu().numpy().reshape(-1))
        logits_all.append(logits.detach().cpu().numpy())
        probs_all.append(probs.detach().cpu().numpy())

    return (
        np.concatenate(y_all, axis=0),
        np.concatenate(pred_all, axis=0),
        np.concatenate(logits_all, axis=0),
        np.concatenate(probs_all, axis=0),
    )

def train_student_distilled(
    student: StudentClassifier,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    *,
    age_grid: torch.Tensor,
    device: torch.device,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    lambda_kd: float,
    rkd_scale: float,
    use_residual_bag: bool,
    kd_method: str = "rkd",
    distill_temperature: float = 2.0,
    verbose: bool,
    tag: str,
) -> TrainResult:
    """
    Train the student with RKD-on-BAGs distillation.
    The KD term is rkd_scale * RKD_dist(student_BAGs, teacher_BAGs); see
    models_v2.combined_student_classification_loss. lambda_kd is the convex
    combination weight on the *scaled* KD term, so it remains in [0, 1] and
    retains its usual interpretation.
    """
    student.to(device)
    age_grid = age_grid.to(device)
    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=weight_decay)

    best_metric = float("inf")
    best_state = None
    best_epoch = max_epochs
    bad = 0

    hist = {
        "train_loss": [],
        "train_sup_ce": [],
        "train_kd_loss": [],
        "train_sup_weighted": [],
        "train_kd_weighted": [],
        "train_sup_fraction": [],
        "train_kd_fraction": [],
        "train_sup_grad_norm": [],
        "train_kd_grad_norm": [],
        "train_grad_norm_ratio_kd_over_sup": [],
        "train_grad_cosine_sup_vs_kd": [],
        "train_mae": [],
        "train_rmse": [],
        "train_r2": [],
        "train_r": [],
        "val_loss": [],
        "val_sup_ce": [],
        "val_kd_loss": [],
        "val_sup_weighted": [],
        "val_kd_weighted": [],
        "val_sup_fraction": [],
        "val_kd_fraction": [],
        "val_sup_grad_norm": [],
        "val_kd_grad_norm": [],
        "val_grad_norm_ratio_kd_over_sup": [],
        "val_grad_cosine_sup_vs_kd": [],
        "val_mae": [],
        "val_rmse": [],
        "val_r2": [],
        "val_r": [],
    }

    shared_params = get_student_shared_params(student)

    def run_epoch(loader: DataLoader, is_train: bool):
        student.train(is_train)
        total_loss = 0.0
        total_sup = 0.0
        total_kd = 0.0
        total_sup_w = 0.0
        total_kd_w = 0.0
        total_sup_gn = 0.0
        total_kd_gn = 0.0
        total_grad_cos = 0.0
        total_grad_cos_weight = 0.0
        n = 0
        y_all, pred_all = [], []

        for x, y, y_prob, t_logits in loader:
            x = x.to(device)
            y = y.to(device).float().view(-1)
            y_prob = y_prob.to(device).float()
            t_logits = t_logits.to(device).float()

            if is_train:
                opt.zero_grad(set_to_none=True)

            s_logits = student(x)
            loss, l_sup, l_kd = combined_student_classification_loss(
                target_probs=y_prob,
                student_logits=s_logits,
                teacher_logits=t_logits,
                age_grid=age_grid,
                y=y,
                lambda_kd=lambda_kd,
                kd_method=kd_method,
                use_residual_bag=use_residual_bag,
                rkd_scale=rkd_scale,
                distill_temperature=distill_temperature,
                return_parts=True,
            )

            sup_weighted = (1.0 - lambda_kd) * l_sup
            kd_weighted = lambda_kd * l_kd

            if lambda_kd > 0.0:
                sup_grad_norm, kd_grad_norm, grad_cos = compute_two_loss_grad_stats(
                    sup_weighted,
                    kd_weighted,
                    shared_params,
                    keep_graph_for_backward=is_train,
                )
            else:
                # No KD branch -> do not compute a second autograd pass
                sup_grad_norm = 0.0
                kd_grad_norm = 0.0
                grad_cos = float("nan")

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
                opt.step()

            pred_age = logits_to_expected_age(s_logits, age_grid, temperature=1.0)

            bs = y.shape[0]
            total_loss += float(loss.detach().cpu()) * bs
            total_sup += float(l_sup.detach().cpu()) * bs
            total_kd += float(l_kd.detach().cpu()) * bs
            total_sup_w += float(sup_weighted.detach().cpu()) * bs
            total_kd_w += float(kd_weighted.detach().cpu()) * bs
            total_sup_gn += sup_grad_norm * bs
            total_kd_gn += kd_grad_norm * bs
            if np.isfinite(grad_cos):
                total_grad_cos += float(grad_cos) * bs
                total_grad_cos_weight += bs
            n += bs
            y_all.append(y.detach().cpu().numpy().reshape(-1))
            pred_all.append(pred_age.detach().cpu().numpy().reshape(-1))

        denom = max(n, 1)
        y_cat = np.concatenate(y_all, axis=0)
        p_cat = np.concatenate(pred_all, axis=0)
        mae, rmse, r2, r = compute_metrics(y_cat, p_cat)
        grad_cos_epoch = total_grad_cos / total_grad_cos_weight if total_grad_cos_weight > 0 else np.nan

        return (
            total_loss / denom,
            total_sup / denom,
            total_kd / denom,
            total_sup_w / denom,
            total_kd_w / denom,
            total_sup_gn / denom,
            total_kd_gn / denom,
            grad_cos_epoch,
            mae,
            rmse,
            r2,
            r,
        )

    for epoch in range(1, max_epochs + 1):
        tr_loss, tr_sup, tr_kd, tr_sup_w, tr_kd_w, tr_sup_gn, tr_kd_gn, tr_cos, tr_mae, tr_rmse, tr_r2, tr_r = run_epoch(train_loader, is_train=True)
        tr_total_parts = tr_sup_w + tr_kd_w

        hist["train_loss"].append(tr_loss)
        hist["train_sup_ce"].append(tr_sup)
        hist["train_kd_loss"].append(tr_kd)
        hist["train_sup_weighted"].append(tr_sup_w)
        hist["train_kd_weighted"].append(tr_kd_w)
        hist["train_sup_fraction"].append(tr_sup_w / tr_total_parts if tr_total_parts > 0 else np.nan)
        hist["train_kd_fraction"].append(tr_kd_w / tr_total_parts if tr_total_parts > 0 else np.nan)
        hist["train_sup_grad_norm"].append(tr_sup_gn)
        hist["train_kd_grad_norm"].append(tr_kd_gn)
        hist["train_grad_norm_ratio_kd_over_sup"].append(tr_kd_gn / tr_sup_gn if tr_sup_gn > 0 else np.nan)
        hist["train_grad_cosine_sup_vs_kd"].append(tr_cos)
        hist["train_mae"].append(tr_mae)
        hist["train_rmse"].append(tr_rmse)
        hist["train_r2"].append(tr_r2)
        hist["train_r"].append(tr_r)

        if val_loader is not None:
            va_loss, va_sup, va_kd, va_sup_w, va_kd_w, va_sup_gn, va_kd_gn, va_cos, va_mae, va_rmse, va_r2, va_r = run_epoch(val_loader, is_train=False)
            va_total_parts = va_sup_w + va_kd_w
            hist["val_loss"].append(va_loss)
            hist["val_sup_ce"].append(va_sup)
            hist["val_kd_loss"].append(va_kd)
            hist["val_sup_weighted"].append(va_sup_w)
            hist["val_kd_weighted"].append(va_kd_w)
            hist["val_sup_fraction"].append(va_sup_w / va_total_parts if va_total_parts > 0 else np.nan)
            hist["val_kd_fraction"].append(va_kd_w / va_total_parts if va_total_parts > 0 else np.nan)
            hist["val_sup_grad_norm"].append(va_sup_gn)
            hist["val_kd_grad_norm"].append(va_kd_gn)
            hist["val_grad_norm_ratio_kd_over_sup"].append(va_kd_gn / va_sup_gn if va_sup_gn > 0 else np.nan)
            hist["val_grad_cosine_sup_vs_kd"].append(va_cos)
            hist["val_mae"].append(va_mae)
            hist["val_rmse"].append(va_rmse)
            hist["val_r2"].append(va_r2)
            hist["val_r"].append(va_r)
            current_metric = va_loss
        else:
            hist["val_loss"].append(np.nan)
            hist["val_sup_ce"].append(np.nan)
            hist["val_kd_loss"].append(np.nan)
            hist["val_sup_weighted"].append(np.nan)
            hist["val_kd_weighted"].append(np.nan)
            hist["val_sup_fraction"].append(np.nan)
            hist["val_kd_fraction"].append(np.nan)
            hist["val_sup_grad_norm"].append(np.nan)
            hist["val_kd_grad_norm"].append(np.nan)
            hist["val_grad_norm_ratio_kd_over_sup"].append(np.nan)
            hist["val_grad_cosine_sup_vs_kd"].append(np.nan)
            hist["val_mae"].append(np.nan)
            hist["val_rmse"].append(np.nan)
            hist["val_r2"].append(np.nan)
            hist["val_r"].append(np.nan)
            va_loss = va_sup_w = va_kd_w = va_sup_gn = va_kd_gn = va_cos = np.nan
            current_metric = tr_loss

        if verbose:
            print(
                f"[{tag}] epoch {epoch:03d} | "
                f"train loss {tr_loss:.4f} sup_w {tr_sup_w:.4f} kd_w {tr_kd_w:.4f} "
                f"sup_gn {tr_sup_gn:.4f} kd_gn {tr_kd_gn:.4f} cos {tr_cos:.4f} | "
                f"val loss {va_loss:.4f} sup_w {va_sup_w:.4f} kd_w {va_kd_w:.4f} "
                f"sup_gn {va_sup_gn:.4f} kd_gn {va_kd_gn:.4f} cos {va_cos:.4f}"
            )

        if val_loader is not None:
            if current_metric < best_metric - 1e-6:
                best_metric = current_metric
                best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
                best_epoch = epoch
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    if verbose:
                        print(f"[{tag}] early stopping at epoch {epoch} (best epoch {best_epoch}, best val loss {best_metric:.4f})")
                    break
        else:
            best_epoch = max_epochs

    if val_loader is not None and best_state is not None:
        student.load_state_dict(best_state)

    return TrainResult(history=hist, best_epoch=best_epoch)

@torch.no_grad()
def predict_student(
    model: StudentClassifier,
    loader: DataLoader,
    *,
    age_grid: torch.Tensor,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    age_grid = age_grid.to(device)
    y_all, pred_all, logits_all, probs_all = [], [], [], []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device).float().view(-1)

        logits = model(x)
        probs = logits_to_probs(logits, temperature=1.0)
        pred_age = logits_to_expected_age(logits, age_grid, temperature=1.0)

        y_all.append(y.detach().cpu().numpy().reshape(-1))
        pred_all.append(pred_age.detach().cpu().numpy().reshape(-1))
        logits_all.append(logits.detach().cpu().numpy())
        probs_all.append(probs.detach().cpu().numpy())

    return (
        np.concatenate(y_all, axis=0),
        np.concatenate(pred_all, axis=0),
        np.concatenate(logits_all, axis=0),
        np.concatenate(probs_all, axis=0),
    )

# -----------------------------------------------------------------------------
# OOF / cross-fit helpers
# -----------------------------------------------------------------------------
def teacher_oof_logits_on_train_with_early_stopping(
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    *,
    conf_train: pd.DataFrame,
    teacher_modality: str,
    teacher_feat_cols: List[str],
    site_cols: List[str],
    age_grid_np: np.ndarray,
    label_sigma_t: float,
    teacher_cfg: GDModelConfig,
    device: torch.device,
    inner_folds: int,
    inner_seed: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    strata_train: np.ndarray,
) -> np.ndarray:
    n_classes = len(age_grid_np)
    logits_oof = np.zeros((len(y_train), n_classes), dtype=np.float32)
    skf = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=inner_seed)

    for k, (itr, iva) in enumerate(skf.split(X_train_raw, strata_train), start=1):
        set_seed(inner_seed + 10_000 + k)
        X_tr_raw, X_va_raw = X_train_raw[itr], X_train_raw[iva]
        y_tr, y_va = y_train[itr], y_train[iva]
        conf_tr = conf_train.iloc[itr].reset_index(drop=True)
        conf_va = conf_train.iloc[iva].reset_index(drop=True)

        dec = FoldDeconfounder(teacher_modality, teacher_feat_cols, site_cols=site_cols).fit(X_tr_raw, conf_tr)
        X_tr = dec.transform(X_tr_raw, conf_tr)
        X_va = dec.transform(X_va_raw, conf_va)
        yprob_tr = make_soft_targets_np(y_tr, age_grid_np, sigma=label_sigma_t)
        yprob_va = make_soft_targets_np(y_va, age_grid_np, sigma=label_sigma_t)

        tr_loader = make_soft_loader(X_tr, y_tr, yprob_tr, batch_size, True, drop_last=_safe_drop_last(len(y_tr), batch_size))
        va_loader = make_soft_loader(X_va, y_va, yprob_va, batch_size, False)
        va_eval_loader = make_eval_loader(X_va, y_va, batch_size, False)

        teacher = TeacherClassifier(
            x_dim=X_tr.shape[1],
            n_classes=n_classes,
            cfg=teacher_cfg,
        )
        train_supervised_classifier(
            teacher,
            tr_loader,
            va_loader,
            age_grid=torch.from_numpy(age_grid_np.astype(np.float32)),
            device=device,
            lr=lr,
            weight_decay=weight_decay,
            max_epochs=max_epochs,
            patience=patience,
            verbose=False,
            tag=f"T-OOF{k}",
        )
        _, _, logits_va, _ = predict_teacher(
            teacher,
            va_eval_loader,
            age_grid=torch.from_numpy(age_grid_np.astype(np.float32)),
            device=device,
        )
        logits_oof[iva] = logits_va.astype(np.float32)

    return logits_oof

def teacher_crossfit_logits_and_pred_fixed(
    X_raw: np.ndarray,
    y: np.ndarray,
    *,
    conf_all: pd.DataFrame,
    teacher_modality: str,
    teacher_feat_cols: List[str],
    site_cols: List[str],
    age_grid_np: np.ndarray,
    label_sigma_t: float,
    teacher_cfg: GDModelConfig,
    device: torch.device,
    crossfit_folds: int,
    crossfit_seed: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    fixed_epochs: int,
    strata: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    n_classes = len(age_grid_np)
    logits_oof = np.zeros((len(y), n_classes), dtype=np.float32)
    pred_oof = np.zeros(len(y), dtype=np.float32)
    skf = StratifiedKFold(n_splits=crossfit_folds, shuffle=True, random_state=crossfit_seed)

    for k, (itr, iva) in enumerate(skf.split(X_raw, strata), start=1):
        set_seed(crossfit_seed + 100_000 + k)
        X_tr_raw, X_va_raw = X_raw[itr], X_raw[iva]
        y_tr, y_va = y[itr], y[iva]
        conf_tr = conf_all.iloc[itr].reset_index(drop=True)
        conf_va = conf_all.iloc[iva].reset_index(drop=True)

        dec = FoldDeconfounder(teacher_modality, teacher_feat_cols, site_cols=site_cols).fit(X_tr_raw, conf_tr)
        X_tr = dec.transform(X_tr_raw, conf_tr)
        X_va = dec.transform(X_va_raw, conf_va)
        yprob_tr = make_soft_targets_np(y_tr, age_grid_np, sigma=label_sigma_t)

        tr_loader = make_soft_loader(X_tr, y_tr, yprob_tr, batch_size, True)
        va_eval_loader = make_eval_loader(X_va, y_va, batch_size, False)

        teacher = TeacherClassifier(
            x_dim=X_tr.shape[1],
            n_classes=n_classes,
            cfg=teacher_cfg,
        )
        train_supervised_classifier(
            teacher,
            tr_loader,
            val_loader=None,
            age_grid=torch.from_numpy(age_grid_np.astype(np.float32)),
            device=device,
            lr=lr,
            weight_decay=weight_decay,
            max_epochs=fixed_epochs,
            patience=fixed_epochs,
            verbose=False,
            tag=f"T-CROSSFIT{k}",
        )
        _, pred_va, logits_va, _ = predict_teacher(
            teacher,
            va_eval_loader,
            age_grid=torch.from_numpy(age_grid_np.astype(np.float32)),
            device=device,
        )
        logits_oof[iva] = logits_va.astype(np.float32)
        pred_oof[iva] = pred_va.astype(np.float32)

    return logits_oof, pred_oof

def student_crossfit_pred_fixed(
    X_raw: np.ndarray,
    y: np.ndarray,
    *,
    teacher_feature_raw: np.ndarray,
    conf_all: pd.DataFrame,
    teacher_modality: str,
    teacher_feat_cols: List[str],
    student_modality: str,
    student_feat_cols: List[str],
    site_cols: List[str],
    age_grid_np: np.ndarray,
    label_sigma_t: float,
    label_sigma_s: float,
    teacher_cfg: GDModelConfig,
    student_cfg: GDModelConfig,
    device: torch.device,
    crossfit_folds: int,
    crossfit_seed: int,
    inner_folds: int,
    batch_size: int,
    teacher_lr: float,
    student_lr: float,
    teacher_weight_decay: float,
    student_weight_decay: float,
    teacher_fixed_epochs: int,
    student_fixed_epochs: int,
    lambda_kd: float,
    rkd_scale: float,
    use_residual_bag: bool,
    kd_method: str = "rkd",
    distill_temperature: float = 2.0,
    strata: np.ndarray,
) -> np.ndarray:
    pred_oof = np.zeros(len(y), dtype=np.float32)
    outer = StratifiedKFold(n_splits=crossfit_folds, shuffle=True, random_state=crossfit_seed)

    for k, (itr, iva) in enumerate(outer.split(X_raw, strata), start=1):
        set_seed(crossfit_seed + 300_000 + k)
        X_tr_raw, X_va_raw = X_raw[itr], X_raw[iva]
        y_tr, y_va = y[itr], y[iva]
        conf_tr = conf_all.iloc[itr].reset_index(drop=True)
        conf_va = conf_all.iloc[iva].reset_index(drop=True)

        Xt_tr_raw = teacher_feature_raw[itr]

        # Leakage-aware teacher OOF logits inside the cross-fit training subset.
        # Skip entirely at lambda_kd == 0: the KD loss short-circuits and the
        # logits are never used, so this avoids wasted compute and keeps the
        # RNG state aligned with baseline_crossfit_pred_fixed.
        if lambda_kd > 0.0:
            strata_tr = strata[itr]
            teacher_logits_tr_oof = teacher_oof_logits_on_train_with_early_stopping(
                X_train_raw=Xt_tr_raw,
                y_train=y_tr,
                conf_train=conf_tr,
                teacher_modality=teacher_modality,
                teacher_feat_cols=teacher_feat_cols,
                site_cols=site_cols,
                age_grid_np=age_grid_np,
                label_sigma_t=label_sigma_t,
                teacher_cfg=teacher_cfg,
                device=device,
                inner_folds=inner_folds,
                inner_seed=crossfit_seed + 10000 + k,
                batch_size=batch_size,
                lr=teacher_lr,
                weight_decay=teacher_weight_decay,
                max_epochs=teacher_fixed_epochs,
                patience=teacher_fixed_epochs,
                strata_train=strata_tr,
            )
            set_seed(crossfit_seed + 300_000 + k)
        else:
            teacher_logits_tr_oof = np.zeros((len(y_tr), len(age_grid_np)), dtype=np.float32)

        yprob_tr = make_soft_targets_np(y_tr, age_grid_np, sigma=label_sigma_s)
        dec = FoldDeconfounder(student_modality, student_feat_cols, site_cols=site_cols).fit(X_tr_raw, conf_tr)
        X_tr = dec.transform(X_tr_raw, conf_tr)
        X_va = dec.transform(X_va_raw, conf_va)

        # drop_last on the RKD training loader to avoid degenerate tiny final batches.
        s_drop_last = _safe_drop_last(len(y_tr), batch_size)
        tr_loader = make_kd_loader(
            X_tr, y_tr, yprob_tr, teacher_logits_tr_oof,
            batch_size, True, drop_last=s_drop_last,
        )
        va_eval_loader = make_eval_loader(X_va, y_va, batch_size, False)
        set_seed(crossfit_seed + 300_000 + k)
        student = StudentClassifier(
            x_dim=X_tr.shape[1],
            n_classes=len(age_grid_np),
            cfg=student_cfg,
        )
        train_student_distilled(
            student,
            tr_loader,
            val_loader=None,
            age_grid=torch.from_numpy(age_grid_np.astype(np.float32)),
            device=device,
            lr=student_lr,
            weight_decay=student_weight_decay,
            max_epochs=student_fixed_epochs,
            patience=student_fixed_epochs,
            lambda_kd=lambda_kd,
            rkd_scale=rkd_scale,
            use_residual_bag=use_residual_bag,
            kd_method=kd_method,
            distill_temperature=distill_temperature,
            verbose=False,
            tag=f"S-CROSSFIT{k}",
        )

        _, pred_va, _, _ = predict_student(
            student,
            va_eval_loader,
            age_grid=torch.from_numpy(age_grid_np.astype(np.float32)),
            device=device,
        )
        pred_oof[iva] = pred_va.astype(np.float32)

    return pred_oof

def baseline_crossfit_pred_fixed(
    X_raw: np.ndarray,
    y: np.ndarray,
    *,
    conf_all: pd.DataFrame,
    student_modality: str,
    student_feat_cols: List[str],
    site_cols: List[str],
    age_grid_np: np.ndarray,
    label_sigma_s: float,
    baseline_cfg: GDModelConfig,
    device: torch.device,
    crossfit_folds: int,
    crossfit_seed: int,
    batch_size: int,
    baseline_lr: float,
    weight_decay: float,
    baseline_fixed_epochs: int,
    strata: np.ndarray,
) -> np.ndarray:
    pred_oof = np.zeros(len(y), dtype=np.float32)
    outer = StratifiedKFold(n_splits=crossfit_folds, shuffle=True, random_state=crossfit_seed)

    for k, (itr, iva) in enumerate(outer.split(X_raw, strata), start=1):
        set_seed(crossfit_seed + 300_000 + k)
        X_tr_raw, X_va_raw = X_raw[itr], X_raw[iva]
        y_tr, y_va = y[itr], y[iva]
        conf_tr = conf_all.iloc[itr].reset_index(drop=True)
        conf_va = conf_all.iloc[iva].reset_index(drop=True)

        yprob_tr = make_soft_targets_np(y_tr, age_grid_np, sigma=label_sigma_s)
        dec = FoldDeconfounder(student_modality, student_feat_cols, site_cols=site_cols).fit(X_tr_raw, conf_tr)
        X_tr = dec.transform(X_tr_raw, conf_tr)
        X_va = dec.transform(X_va_raw, conf_va)

        tr_loader = make_soft_loader(X_tr, y_tr, yprob_tr, batch_size, True, drop_last=_safe_drop_last(len(y_tr), batch_size))
        va_eval_loader = make_eval_loader(X_va, y_va, batch_size, False)

        baseline = make_baseline_model(
            x_dim=X_tr.shape[1],
            n_classes=len(age_grid_np),
            cfg=baseline_cfg,
        )
        train_supervised_classifier(
            baseline,
            tr_loader,
            val_loader=None,
            age_grid=torch.from_numpy(age_grid_np.astype(np.float32)),
            device=device,
            lr=baseline_lr,
            weight_decay=weight_decay,
            max_epochs=baseline_fixed_epochs,
            patience=baseline_fixed_epochs,
            verbose=False,
            tag=f"BL-CROSSFIT{k}",
        )

        _, pred_va, _, _ = predict_student(
            baseline,
            va_eval_loader,
            age_grid=torch.from_numpy(age_grid_np.astype(np.float32)),
            device=device,
        )
        pred_oof[iva] = pred_va.astype(np.float32)

    return pred_oof

# -----------------------------------------------------------------------------
# Final training helpers
# -----------------------------------------------------------------------------
def train_final_teacher_on_all_dev(
    X_raw: np.ndarray,
    y: np.ndarray,
    *,
    conf_all: pd.DataFrame,
    teacher_modality: str,
    teacher_feat_cols: List[str],
    site_cols: List[str],
    age_grid_np: np.ndarray,
    label_sigma_t: float,
    teacher_cfg: GDModelConfig,
    device: torch.device,
    batch_size: int,
    lr: float,
    weight_decay: float,
    n_epochs: int,
) -> Tuple[TeacherClassifier, FoldDeconfounder, Dict[str, List[float]]]:
    dec = FoldDeconfounder(teacher_modality, teacher_feat_cols, site_cols=site_cols).fit(X_raw, conf_all)
    X = dec.transform(X_raw, conf_all)
    y_prob = make_soft_targets_np(y, age_grid_np, sigma=label_sigma_t)
    loader = make_soft_loader(X, y, y_prob, batch_size, True)

    teacher = TeacherClassifier(
        x_dim=X.shape[1],
        n_classes=len(age_grid_np),
        cfg=teacher_cfg,
    )
    tr = train_supervised_classifier(
        teacher,
        loader,
        val_loader=None,
        age_grid=torch.from_numpy(age_grid_np.astype(np.float32)),
        device=device,
        lr=lr,
        weight_decay=weight_decay,
        max_epochs=n_epochs,
        patience=n_epochs,
        verbose=False,
        tag="T-FINAL",
    )
    return teacher, dec, tr.history

def train_final_student_on_all_dev(
    X_raw: np.ndarray,
    y: np.ndarray,
    teacher_logits_oof: np.ndarray,
    *,
    conf_all: pd.DataFrame,
    student_modality: str,
    student_feat_cols: List[str],
    site_cols: List[str],
    age_grid_np: np.ndarray,
    label_sigma_s: float,
    student_cfg: GDModelConfig,
    device: torch.device,
    batch_size: int,
    lr: float,
    weight_decay: float,
    n_epochs: int,
    lambda_kd: float,
    rkd_scale: float,
    use_residual_bag: bool,
    kd_method: str = "rkd",
    distill_temperature: float = 2.0,
) -> Tuple[StudentClassifier, FoldDeconfounder, Dict[str, List[float]]]:
    feat_prep = FoldDeconfounder(student_modality, student_feat_cols, site_cols=site_cols).fit(X_raw, conf_all)
    X = feat_prep.transform(X_raw, conf_all)
    y_prob = make_soft_targets_np(y, age_grid_np, sigma=label_sigma_s)
    s_drop_last = _safe_drop_last(len(y), batch_size)
    loader = make_kd_loader(
        X, y, y_prob, teacher_logits_oof, batch_size, True,
        drop_last=s_drop_last,
    )

    student = StudentClassifier(
        x_dim=X.shape[1],
        n_classes=len(age_grid_np),
        cfg=student_cfg,
    )
    tr = train_student_distilled(
        student,
        loader,
        val_loader=None,
        age_grid=torch.from_numpy(age_grid_np.astype(np.float32)),
        device=device,
        lr=lr,
        weight_decay=weight_decay,
        max_epochs=n_epochs,
        patience=n_epochs,
        lambda_kd=lambda_kd,
        rkd_scale=rkd_scale,
        use_residual_bag=use_residual_bag,
        kd_method=kd_method,
        distill_temperature=distill_temperature,
        verbose=False,
        tag="S-FINAL",
    )
    return student, feat_prep, tr.history

def train_final_baseline_on_all_dev(
    X_raw: np.ndarray,
    y: np.ndarray,
    *,
    conf_all: pd.DataFrame,
    student_modality: str,
    student_feat_cols: List[str],
    site_cols: List[str],
    age_grid_np: np.ndarray,
    label_sigma_s: float,
    baseline_cfg: GDModelConfig,
    device: torch.device,
    batch_size: int,
    lr: float,
    weight_decay: float,
    n_epochs: int,
) -> Tuple[torch.nn.Module, FoldDeconfounder, Dict[str, List[float]]]:
    feat_prep = FoldDeconfounder(student_modality, student_feat_cols, site_cols=site_cols).fit(X_raw, conf_all)
    X = feat_prep.transform(X_raw, conf_all)
    y_prob = make_soft_targets_np(y, age_grid_np, sigma=label_sigma_s)
    loader = make_soft_loader(X, y, y_prob, batch_size, True, drop_last=_safe_drop_last(len(y), batch_size))

    baseline = make_baseline_model(
        x_dim=X.shape[1],
        n_classes=len(age_grid_np),
        cfg=baseline_cfg,
    )
    tr = train_supervised_classifier(
        baseline,
        loader,
        val_loader=None,
        age_grid=torch.from_numpy(age_grid_np.astype(np.float32)),
        device=device,
        lr=lr,
        weight_decay=weight_decay,
        max_epochs=n_epochs,
        patience=n_epochs,
        verbose=False,
        tag="BL-FINAL",
    )
    return baseline, feat_prep, tr.history

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--smri_csv", type=str, required=True)
    ap.add_argument("--reg_csv", type=str, required=True)
    ap.add_argument("--target_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="gd_hc_dev_cv_plus_normative_holdout_out")

    ap.add_argument("--eid_col", type=str, default="eid")
    ap.add_argument("--age_col", type=str, default="age_at_assessment")
    ap.add_argument("--hc_col", type=str, required=True)
    ap.add_argument("--hc_value", type=int, default=1)

    ap.add_argument("--sex_col", type=str, default="sex")
    ap.add_argument("--site_prefix", type=str, default="site_")

    # Confound correction: modality of each block decides its confound set.
    #   smri -> volumes on eTIV+sex+site, thickness on sex+site
    #   fmri -> across-network connectivity on motion+sex+site
    #   questionnaire -> z-score only (no residualization)
    ap.add_argument("--teacher_modality", type=str, default="smri",
                    choices=["smri", "fmri", "questionnaire"],
                    help="Modality of the teacher feature block; selects its confound set.")
    ap.add_argument("--student_modality", type=str, default="questionnaire",
                    choices=["smri", "fmri", "questionnaire"],
                    help="Modality of the student/baseline feature block; selects its confound set.")
    ap.add_argument("--deconfound", action="store_true",
                    help="Enable leakage-safe per-fold feature-level confound residualization "
                         "(fit on train, applied forward). If off, features are z-scored only.")

    # NOTE: These columns are no longer fed into any model. They are held OUT of
    # the brain-feature matrices (so feature composition is unchanged) and rows
    # missing them are still dropped, so they remain available for DOWNSTREAM
    # confounder correction of the BAG-outcome associations.
    ap.add_argument("--teacher_cov_cols", type=str, default="sex,EstimatedTotalIntraCranialVol,MeanHeadMotionrfMRI",
                    help="Columns excluded from the teacher (sMRI) feature matrix and reserved for downstream confounder correction.")
    ap.add_argument("--teacher_cov_prefixes", type=str, default="site_",
                    help="Column prefixes excluded from the teacher feature matrix (e.g. site dummies).")
    ap.add_argument("--student_cov_cols", type=str, default="sex,EstimatedTotalIntraCranialVol",
                    help="Columns excluded from the student (regular) feature matrix and reserved for downstream confounder correction.")
    ap.add_argument("--student_cov_prefixes", type=str, default="site_",
                    help="Column prefixes excluded from the student feature matrix (e.g. site dummies).")

    ap.add_argument("--norm_holdout_frac", type=float, default=0.5, help="Fraction of HC reserved as untouched normative-reference holdout.")
    ap.add_argument("--k_folds", type=int, default=10, help="Outer CV folds inside HC development set and cross-fit folds on HC development.")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--age_bins", type=int, default=10)
    ap.add_argument("--stratify_site", action="store_true")
    ap.add_argument("--stratify_sex", action="store_true")
    ap.add_argument("--inner_folds", type=int, default=5)

    ap.add_argument("--age_bin_step", type=float, default=1.0)
    ap.add_argument("--age_grid_margin_bins", type=int, default=1)
    ap.add_argument(
        "--label_sigma",
        type=float,
        default=1.0,
        help="Gaussian soft-label sigma for the STUDENT and BASELINE supervision targets (label_sigma_s).",
    )
    ap.add_argument(
        "--label_sigma_t",
        type=float,
        default=None,
        help=(
            "Gaussian soft-label sigma for the TEACHER supervision targets. "
            "If not provided, defaults to --label_sigma (previous single-sigma behavior)."
        ),
    )

    # KD: Relational KD on BAGs.
    ap.add_argument(
        "--rkd_scale",
        type=float,
        default=50.0,
        help=(
            "Multiplicative scale on the raw RKD distance loss. The KD term is "
            "rkd_scale * RKD_dist(student_BAGs, teacher_BAGs). Default 50.0 makes "
            "the supervised CE and KD contributions roughly comparable so that "
            "lambda_kd retains its convex-combination interpretation."
        ),
    )
    ap.add_argument(
        "--rkd_target",
        type=str,
        default="bag",
        choices=["bag", "expected_age"],
        help=(
            "What scalar to compute pairwise RKD distances on: 'bag' (residual = "
            "predicted age - chronological age, recommended for transferring "
            "privileged-modality signal beyond age) or 'expected_age'."
        ),
    )
    ap.add_argument(
        "--kd_method",
        type=str,
        default="rkd",
        choices=["rkd", "soft_label"],
        help=(
            "KD loss variant. 'rkd': Relational KD on BAGs (Park et al., 2019). "
            "'soft_label': output-distribution matching (Hinton et al., 2015). "
            "Use 'soft_label' for cross-domain modality pairs (e.g. sMRI->questionnaires) "
            "where RKD produces systematically negative gradient cosines."
        ),
    )
    ap.add_argument(
        "--distill_temperature",
        type=float,
        default=2.0,
        help=(
            "[soft_label only] Temperature for softening teacher/student distributions. "
            "Higher T spreads probability mass to neighboring age bins. Ignored for RKD."
        ),
    )

    ap.add_argument("--activation", type=str, default="gelu")
    ap.add_argument("--dropout", type=float, default=0.2,
                    help="Fallback dropout if --teacher_dropout/--student_dropout/--baseline_dropout are not given.")
    ap.add_argument("--teacher_dropout", type=float, default=None,
                    help="Teacher dropout (overrides --dropout for the teacher cfg). From tune.py teacher stage.")
    ap.add_argument("--student_dropout", type=float, default=None,
                    help="Student dropout (overrides --dropout for the student cfg). From tune.py student stage.")
    ap.add_argument("--baseline_dropout", type=float, default=None,
                    help="Baseline dropout (overrides --dropout for the baseline cfg). Set equal to student for the matched comparison.")
    ap.add_argument("--no_layer_norm", action="store_true")
    ap.add_argument("--teacher_rep_hidden", type=str, default="256,128")
    ap.add_argument("--teacher_rep_dim", type=int, default=128)
    ap.add_argument("--teacher_head_hidden", type=str, default="128")
    ap.add_argument("--student_rep_hidden", type=str, default="128,64")
    ap.add_argument("--student_rep_dim", type=int, default=64)
    ap.add_argument("--student_head_hidden", type=str, default="64")

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--teacher_lr", type=float, default=1e-3)
    ap.add_argument("--student_lr", type=float, default=1e-2)
    ap.add_argument("--baseline_lr", type=float, default=1e-2)
    ap.add_argument("--weight_decay", type=float, default=1e-4,
                    help="Fallback weight decay if --teacher/student/baseline_weight_decay are not given.")
    ap.add_argument("--teacher_weight_decay", type=float, default=None,
                    help="Teacher weight decay (overrides --weight_decay). From tune.py teacher stage.")
    ap.add_argument("--student_weight_decay", type=float, default=None,
                    help="Student weight decay (overrides --weight_decay). From tune.py student stage.")
    ap.add_argument("--baseline_weight_decay", type=float, default=None,
                    help="Baseline weight decay (overrides --weight_decay). Set equal to student for the matched comparison.")
    ap.add_argument("--max_epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--lambda_kd", type=float, required=True, help="Convex weight on the RKD-scaled KD term in [0, 1].")

    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--verbose", action="store_true")

    args = ap.parse_args()

    if not (0.0 < args.norm_holdout_frac < 1.0):
        raise ValueError("--norm_holdout_frac must be strictly between 0 and 1.")

    label_sigma_s = float(args.label_sigma)
    label_sigma_t = float(args.label_sigma_t) if args.label_sigma_t is not None else label_sigma_s
    if label_sigma_s <= 0.0:
        raise ValueError("--label_sigma (student/baseline sigma) must be strictly positive.")
    if label_sigma_t <= 0.0:
        raise ValueError("--label_sigma_t (teacher sigma) must be strictly positive.")
    if args.rkd_scale <= 0.0:
        raise ValueError("--rkd_scale must be strictly positive.")
    if not (0.0 <= args.lambda_kd <= 1.0):
        raise ValueError("--lambda_kd must be in [0, 1].")

    use_residual_bag = (args.rkd_target == "bag")
    # Build a descriptive tag for output CSVs
    if args.kd_method == "rkd":
        kd_method_tag = f"rkd_distance_{'residual_bag' if use_residual_bag else 'expected_age'}"
    else:
        kd_method_tag = f"soft_label_T{args.distill_temperature}"

    print(f"KD method            : {kd_method_tag}")
    print(f"Distill temperature  : {args.distill_temperature}" if args.kd_method == "soft_label" else "")
    print(f"RKD target           : {args.rkd_target} (use_residual_bag={use_residual_bag})")
    print(f"RKD scale            : {args.rkd_scale}")
    print(f"Student lambda_kd    : {args.lambda_kd}")
    print(f"Label sigma (teacher): {label_sigma_t}")
    print(f"Label sigma (student): {label_sigma_s}")
    print(f"norm_holdout_frac    : {args.norm_holdout_frac}")

    set_seed(args.seed)
    ensure_dir(args.out_dir)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print(f"Using device: {device}")

    layer_norm = not args.no_layer_norm
    teacher_dropout = args.teacher_dropout if args.teacher_dropout is not None else args.dropout
    student_dropout = args.student_dropout if args.student_dropout is not None else args.dropout
    baseline_dropout = args.baseline_dropout if args.baseline_dropout is not None else args.dropout
    teacher_weight_decay = args.teacher_weight_decay if args.teacher_weight_decay is not None else args.weight_decay
    student_weight_decay = args.student_weight_decay if args.student_weight_decay is not None else args.weight_decay
    baseline_weight_decay = args.baseline_weight_decay if args.baseline_weight_decay is not None else args.weight_decay
    print(f"Dropout      | teacher={teacher_dropout} | student={student_dropout} | baseline={baseline_dropout}")
    print(f"Weight decay | teacher={teacher_weight_decay} | student={student_weight_decay} | baseline={baseline_weight_decay}")
    teacher_cfg = GDModelConfig(
        activation=args.activation,
        dropout=teacher_dropout,
        layer_norm=layer_norm,
        rep_hidden=parse_int_tuple(args.teacher_rep_hidden),
        rep_dim=args.teacher_rep_dim,
        head_hidden=parse_int_tuple(args.teacher_head_hidden),
    )
    student_cfg = GDModelConfig(
        activation=args.activation,
        dropout=student_dropout,
        layer_norm=layer_norm,
        rep_hidden=parse_int_tuple(args.student_rep_hidden),
        rep_dim=args.student_rep_dim,
        head_hidden=parse_int_tuple(args.student_head_hidden),
    )
    # Baseline mirrors the student architecture exactly, but with its own dropout.
    baseline_cfg = GDModelConfig(
        activation=args.activation,
        dropout=baseline_dropout,
        layer_norm=layer_norm,
        rep_hidden=parse_int_tuple(args.student_rep_hidden),
        rep_dim=args.student_rep_dim,
        head_hidden=parse_int_tuple(args.student_head_hidden),
    )

    smri = pd.read_csv(args.smri_csv)
    reg = pd.read_csv(args.reg_csv)
    tgt = pd.read_csv(args.target_csv)

    assert_unique_eids(smri, name="smri", eid_col=args.eid_col)
    assert_unique_eids(reg, name="reg", eid_col=args.eid_col)
    assert_unique_eids(tgt, name="target", eid_col=args.eid_col)

    df = tgt.merge(smri, on=args.eid_col, how="inner").merge(reg, on=args.eid_col, how="inner")

    assert_unique_eids(df, name="merged dataframe", eid_col=args.eid_col)
    print(f"Merged N={len(df)}")

    if args.age_col not in df.columns:
        raise ValueError(f"age_col='{args.age_col}' not found in merged data.")
    if args.hc_col not in df.columns:
        raise ValueError(f"hc_col='{args.hc_col}' not found in merged data.")

    # Columns held out of the feature matrices and reserved for downstream
    # confounder correction (they never enter any model).
    teacher_excl_cols = [c for c in parse_cols(args.teacher_cov_cols) if c in df.columns]
    teacher_excl_cols += cols_by_prefix(df, parse_cols(args.teacher_cov_prefixes))
    teacher_excl_cols = uniq(teacher_excl_cols)

    student_excl_cols = [c for c in parse_cols(args.student_cov_cols) if c in df.columns]
    student_excl_cols += cols_by_prefix(df, parse_cols(args.student_cov_prefixes))
    student_excl_cols = uniq(student_excl_cols)

    smri_numeric = [c for c in smri.columns if c != args.eid_col and c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    reg_numeric = [c for c in reg.columns if c != args.eid_col and c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

    smri_feat_cols = [c for c in smri_numeric if c not in set(teacher_excl_cols)]
    reg_feat_cols = [c for c in reg_numeric if c not in set(student_excl_cols)]

    # Quarantine QC / image-quality columns out of the feature matrices. They are
    # scan-quality proxies (SNR/CNR, registration discrepancy, warping), not brain
    # features; a model can predict age from scan quality, and quality tracks head
    # motion which tracks the downstream outcomes. Kept out of features here; they
    # remain in df for downstream sensitivity analysis.
    smri_feat_cols, smri_qc_cols = split_features_and_qc(smri_feat_cols)
    reg_feat_cols, reg_qc_cols = split_features_and_qc(reg_feat_cols)
    if smri_qc_cols or reg_qc_cols:
        print(f"Quarantined QC columns | teacher-block: {smri_qc_cols} | student-block: {reg_qc_cols}")

    if len(smri_feat_cols) == 0 or len(reg_feat_cols) == 0:
        raise ValueError("Could not infer sMRI and/or regular feature columns after excluding held-out columns.")

    # ---- Confound-correction setup ------------------------------------------
    # Resolve which confound columns exist, and which the chosen modalities need.
    conf_info = resolve_confound_cols(df.columns, site_prefix=args.site_prefix)
    site_cols_conf = conf_info["site_cols"]
    teacher_modality = args.teacher_modality
    student_modality = args.student_modality
    if args.deconfound:
        need_conf = uniq(
            required_confounds_for(teacher_modality, site_cols_conf)
            + required_confounds_for(student_modality, site_cols_conf)
        )
        missing_conf = [c for c in need_conf if c not in df.columns]
        if missing_conf:
            raise ValueError(
                f"--deconfound requested but these confound columns are missing: {missing_conf}. "
                f"Present site dummies: {site_cols_conf}."
            )
        print(f"Deconfounding ON | teacher='{teacher_modality}' student='{student_modality}' "
              f"| confounds used: {need_conf}")
    else:
        # No residualization: FoldDeconfounder still z-scores, but every modality
        # is treated as 'questionnaire' (empty confound set) so behavior matches
        # the previous z-score-only pipeline.
        teacher_modality = "questionnaire"
        student_modality = "questionnaire"
        print("Deconfounding OFF | z-score only (features unchanged vs. previous behavior)")

    # All confound columns that either block might read (union), for slicing.
    conf_cols_all = uniq(
        required_confounds_for(args.teacher_modality, site_cols_conf)
        + required_confounds_for(args.student_modality, site_cols_conf)
    )
    conf_cols_all = [c for c in conf_cols_all if c in df.columns]

    def conf_frame(sub_df: pd.DataFrame) -> pd.DataFrame:
        # Confound frame aligned to sub_df's row order (reset index).
        if conf_cols_all:
            return sub_df[conf_cols_all].reset_index(drop=True)
        return pd.DataFrame(index=range(len(sub_df))).reset_index(drop=True)

    needed = [args.age_col, args.hc_col] + teacher_excl_cols + student_excl_cols + smri_feat_cols + reg_feat_cols
    needed = uniq([c for c in needed if c in df.columns])
    df = df.dropna(subset=needed).copy()
    assert_unique_eids(df, name="post-dropna merged dataframe", eid_col=args.eid_col)
    print(f"After dropping NaNs in age/hc/held-out/features: N={len(df)}")

    hc_df = df[df[args.hc_col] == args.hc_value].copy().reset_index(drop=True)
    case_df = df[df[args.hc_col] != args.hc_value].copy().reset_index(drop=True)
    if len(hc_df) == 0:
        raise ValueError("No healthy controls available after filtering.")

    print(f"HC N={len(hc_df)} | Cases N={len(case_df)}")
    print(f"sMRI features: {len(smri_feat_cols)} | regular features: {len(reg_feat_cols)}")
    print(f"Held-out teacher cols (downstream): {len(teacher_excl_cols)} | student cols: {len(student_excl_cols)}")

    # ------------------------------------------------------------------
    # Stage A0: split HC into development and untouched normative holdout
    # ------------------------------------------------------------------
    hc_strata_full = make_strata(
        hc_df,
        age_col=args.age_col,
        n_age_bins=args.age_bins,
        stratify_site=args.stratify_site,
        site_prefix=args.site_prefix,
        stratify_sex=args.stratify_sex,
        sex_col=args.sex_col,
    )
    sss_norm = StratifiedShuffleSplit(n_splits=1, test_size=args.norm_holdout_frac, random_state=args.seed + 17)
    dev_idx, norm_idx = next(sss_norm.split(np.zeros(len(hc_df)), hc_strata_full))
    hc_dev_df = hc_df.iloc[dev_idx].copy().reset_index(drop=True)
    hc_norm_df = hc_df.iloc[norm_idx].copy().reset_index(drop=True)

    print("\n" + "=" * 90)
    print("HC split:")
    print(f"HC development N={len(hc_dev_df)} | HC normative-reference holdout N={len(hc_norm_df)}")
    log_sex_distribution(hc_dev_df, args.sex_col, "HC-development")
    log_sex_distribution(hc_norm_df, args.sex_col, "HC-normative-holdout")

    # ------------------------------------------------------------------
    # Stage A1: outer CV ONLY within HC development set
    # ------------------------------------------------------------------
    dev_outer_strata = make_strata(
        hc_dev_df,
        age_col=args.age_col,
        n_age_bins=args.age_bins,
        stratify_site=args.stratify_site,
        site_prefix=args.site_prefix,
        stratify_sex=args.stratify_sex,
        sex_col=args.sex_col,
    )
    skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)

    fold_rows: List[Dict[str, Any]] = []
    dev_oof_rows: List[pd.DataFrame] = []
    teacher_best_epochs: List[int] = []
    student_best_epochs: List[int] = []
    baseline_best_epochs: List[int] = []

    tgt_keep_cols = [c for c in tgt.columns if c in hc_df.columns]

    for fold, (trainval_idx, test_idx) in enumerate(skf.split(hc_dev_df, dev_outer_strata), start=1):
        print("\n" + "=" * 90)
        print(f"HC-development outer fold {fold}/{args.k_folds}")

        df_trainval = hc_dev_df.iloc[trainval_idx].copy().reset_index(drop=True)
        df_test = hc_dev_df.iloc[test_idx].copy().reset_index(drop=True)

        age_grid_t = make_age_grid(
            torch.from_numpy(df_trainval[args.age_col].to_numpy().astype(np.float32)),
            age_bin_step=args.age_bin_step,
            margin_bins=args.age_grid_margin_bins,
        )
        age_grid_np = age_grid_t.cpu().numpy().astype(np.float32)
        n_classes = len(age_grid_np)
        print(
            f"Age grid for dev fold {fold}: {age_grid_np[0]:.2f} .. {age_grid_np[-1]:.2f} "
            f"(step={args.age_bin_step}, classes={n_classes})"
        )

        strata_trainval = make_strata(
            df_trainval,
            age_col=args.age_col,
            n_age_bins=args.age_bins,
            stratify_site=args.stratify_site,
            site_prefix=args.site_prefix,
            stratify_sex=args.stratify_sex,
            sex_col=args.sex_col,
        )
        sss_val = StratifiedShuffleSplit(n_splits=1, test_size=args.val_frac, random_state=args.seed + fold)
        tr_rel, va_rel = next(sss_val.split(np.zeros(len(df_trainval)), strata_trainval))
        df_train = df_trainval.iloc[tr_rel].copy().reset_index(drop=True)
        df_val = df_trainval.iloc[va_rel].copy().reset_index(drop=True)

        print(f"HC-dev split sizes | train={len(df_train)} | val={len(df_val)} | test={len(df_test)}")
        log_sex_distribution(df_train, args.sex_col, f"dev_fold{fold} TRAIN-HC")
        log_sex_distribution(df_val, args.sex_col, f"dev_fold{fold} VAL-HC")
        log_sex_distribution(df_test, args.sex_col, f"dev_fold{fold} TEST-HC")

        Xs_train_raw = df_train[smri_feat_cols].to_numpy().astype(np.float32)
        Xs_val_raw = df_val[smri_feat_cols].to_numpy().astype(np.float32)
        Xs_test_raw = df_test[smri_feat_cols].to_numpy().astype(np.float32)

        Xr_train_raw = df_train[reg_feat_cols].to_numpy()
        Xr_val_raw = df_val[reg_feat_cols].to_numpy()
        Xr_test_raw = df_test[reg_feat_cols].to_numpy()

        y_train = df_train[args.age_col].to_numpy().astype(np.float32)
        y_val = df_val[args.age_col].to_numpy().astype(np.float32)
        y_test = df_test[args.age_col].to_numpy().astype(np.float32)

        yprob_train_t = make_soft_targets_np(y_train, age_grid_np, sigma=label_sigma_t)
        yprob_val_t = make_soft_targets_np(y_val, age_grid_np, sigma=label_sigma_t)
        yprob_train_s = make_soft_targets_np(y_train, age_grid_np, sigma=label_sigma_s)
        yprob_val_s = make_soft_targets_np(y_val, age_grid_np, sigma=label_sigma_s)

        conf_train = conf_frame(df_train)
        conf_val = conf_frame(df_val)
        conf_test = conf_frame(df_test)

        teacher_dec = FoldDeconfounder(teacher_modality, smri_feat_cols, site_cols=site_cols_conf).fit(Xs_train_raw, conf_train)
        Xs_train = teacher_dec.transform(Xs_train_raw, conf_train)
        Xs_val = teacher_dec.transform(Xs_val_raw, conf_val)
        Xs_test = teacher_dec.transform(Xs_test_raw, conf_test)

        student_dec = FoldDeconfounder(student_modality, reg_feat_cols, site_cols=site_cols_conf).fit(Xr_train_raw, conf_train)
        Xr_train = student_dec.transform(Xr_train_raw, conf_train)
        Xr_val = student_dec.transform(Xr_val_raw, conf_val)
        Xr_test = student_dec.transform(Xr_test_raw, conf_test)

        # Teacher
        t_train_loader = make_soft_loader(Xs_train, y_train, yprob_train_t, args.batch_size, True)
        t_val_loader = make_soft_loader(Xs_val, y_val, yprob_val_t, args.batch_size, False)
        t_test_loader = make_eval_loader(Xs_test, y_test, args.batch_size, False)
        set_seed(args.seed + 100_000 + fold)
        teacher = TeacherClassifier(
            x_dim=Xs_train.shape[1],
            n_classes=n_classes,
            cfg=teacher_cfg,
        )
        t_train_result = train_supervised_classifier(
            teacher,
            t_train_loader,
            t_val_loader,
            age_grid=age_grid_t,
            device=device,
            lr=args.teacher_lr,
            weight_decay=teacher_weight_decay,
            max_epochs=args.max_epochs,
            patience=args.patience,
            verbose=args.verbose,
            tag=f"T-devfold{fold}",
        )
        teacher_best_epochs.append(t_train_result.best_epoch)

        _, pred_val_t, logits_val_t, _ = predict_teacher(
            teacher,
            make_eval_loader(Xs_val, y_val, args.batch_size, False),
            age_grid=age_grid_t,
            device=device,
        )
        _, pred_test_t, _, probs_test_t = predict_teacher(
            teacher,
            t_test_loader,
            age_grid=age_grid_t,
            device=device,
        )
        t_mae, t_rmse, t_r2, t_r = compute_metrics(y_test, pred_test_t)

        # Teacher OOF logits on HC-dev-train only for student KD training
        strata_train = make_strata(
            df_train,
            age_col=args.age_col,
            n_age_bins=args.age_bins,
            stratify_site=args.stratify_site,
            site_prefix=args.site_prefix,
            stratify_sex=args.stratify_sex,
            sex_col=args.sex_col,
        )
        if float(args.lambda_kd) > 0.0:
            teacher_logits_train_oof = teacher_oof_logits_on_train_with_early_stopping(
                        X_train_raw=Xs_train_raw,
                        y_train=y_train,
                        conf_train=conf_train,
                        teacher_modality=teacher_modality,
                        teacher_feat_cols=smri_feat_cols,
                        site_cols=site_cols_conf,
                        age_grid_np=age_grid_np,
                        label_sigma_t=label_sigma_t,
                        teacher_cfg=teacher_cfg,
                        device=device,
                        inner_folds=args.inner_folds,
                        inner_seed=args.seed + 1000 + fold,
                        batch_size=args.batch_size,
                        lr=args.teacher_lr,
                        weight_decay=teacher_weight_decay,
                        max_epochs=min(args.max_epochs, 120),
                        patience=min(args.patience, 15),
                        strata_train=strata_train,
                    )
            set_seed(args.seed + 300_000 + fold)
        else:
            teacher_logits_train_oof = np.zeros((len(y_train), n_classes), dtype=np.float32)

        # Student
        s_drop_last = _safe_drop_last(len(y_train), args.batch_size)
        s_train_loader = make_kd_loader(
            Xr_train, y_train, yprob_train_s, teacher_logits_train_oof,
            args.batch_size, True, drop_last=_safe_drop_last(len(y_train), args.batch_size),
        )
        s_val_loader = make_kd_loader(
            Xr_val, y_val, yprob_val_s, logits_val_t,
            args.batch_size, False, drop_last=False,
        )
        s_test_loader = make_eval_loader(Xr_test, y_test, args.batch_size, False)
        set_seed(args.seed + 300_000 + fold)
        student = StudentClassifier(
            x_dim=Xr_train.shape[1],
            n_classes=n_classes,
            cfg=student_cfg,
        )
        s_train_result = train_student_distilled(
            student,
            s_train_loader,
            s_val_loader,
            age_grid=age_grid_t,
            device=device,
            lr=args.student_lr,
            weight_decay=student_weight_decay,
            max_epochs=args.max_epochs,
            patience=args.patience,
            lambda_kd=float(args.lambda_kd),
            rkd_scale=float(args.rkd_scale),
            use_residual_bag=use_residual_bag,
            kd_method=args.kd_method,
            distill_temperature=args.distill_temperature,
            verbose=args.verbose,
            tag=f"S-devfold{fold}",
        )
        student_best_epochs.append(s_train_result.best_epoch)

        _, pred_test_s, _, probs_test_s = predict_student(
            student,
            s_test_loader,
            age_grid=age_grid_t,
            device=device,
        )
        s_mae, s_rmse, s_r2, s_r = compute_metrics(y_test, pred_test_s)

        # Baseline
        bl_train_loader = make_soft_loader(Xr_train, y_train, yprob_train_s, args.batch_size, True, drop_last=_safe_drop_last(len(y_train), args.batch_size),)
        bl_val_loader = make_soft_loader(Xr_val, y_val, yprob_val_s, args.batch_size, False)
        bl_test_loader = make_eval_loader(Xr_test, y_test, args.batch_size, False)
        set_seed(args.seed + 300_000 + fold)
        baseline = make_baseline_model(
            x_dim=Xr_train.shape[1],
            n_classes=n_classes,
            cfg=baseline_cfg,
        )
        bl_train_result = train_supervised_classifier(
            baseline,
            bl_train_loader,
            bl_val_loader,
            age_grid=age_grid_t,
            device=device,
            lr=args.baseline_lr,
            weight_decay=baseline_weight_decay,
            max_epochs=args.max_epochs,
            patience=args.patience,
            verbose=args.verbose,
            tag=f"BL-devfold{fold}",
        )
        baseline_best_epochs.append(bl_train_result.best_epoch)

        _, pred_test_bl, _, probs_test_bl = predict_student(
            baseline,
            bl_test_loader,
            age_grid=age_grid_t,
            device=device,
        )
        bl_mae, bl_rmse, bl_r2, bl_r = compute_metrics(y_test, pred_test_bl)

        print("\nHC-development held-out test summary:")
        print(f"Teacher   | MAE {t_mae:.3f} RMSE {t_rmse:.3f} R2 {t_r2:.3f} r {t_r:.3f}")
        print(f"StudentGD | MAE {s_mae:.3f} RMSE {s_rmse:.3f} R2 {s_r2:.3f} r {s_r:.3f}")
        print(f"Baseline  | MAE {bl_mae:.3f} RMSE {bl_rmse:.3f} R2 {bl_r2:.3f} r {bl_r:.3f}")

        fold_dir = os.path.join(args.out_dir, f"hc_dev_fold_{fold:02d}")
        ensure_dir(fold_dir)
        pd.DataFrame(t_train_result.history).to_csv(os.path.join(fold_dir, "teacher_history.csv"), index=False)
        pd.DataFrame(s_train_result.history).to_csv(os.path.join(fold_dir, "student_gd_history.csv"), index=False)
        pd.DataFrame(bl_train_result.history).to_csv(os.path.join(fold_dir, "baseline_history.csv"), index=False)

        pred_df = df_test[tgt_keep_cols].copy()
        pred_df.insert(1, "fold", fold)
        pred_df["set_name"] = "hc_dev_oof"
        pred_df["age_grid_min"] = float(age_grid_np[0])
        pred_df["age_grid_max"] = float(age_grid_np[-1])
        pred_df["age_grid_step"] = float(args.age_bin_step)
        pred_df["label_sigma_t"] = float(label_sigma_t)
        pred_df["label_sigma_s"] = float(label_sigma_s)
        pred_df["kd_method"] = kd_method_tag
        pred_df["rkd_scale"] = float(args.rkd_scale)
        pred_df["rkd_target"] = args.rkd_target
        pred_df["lambda_kd"] = float(args.lambda_kd)

        pred_df["teacher_pred_age_raw"] = pred_test_t
        pred_df["teacher_bag_raw"] = pred_test_t - y_test
        pred_df["teacher_maxprob"] = probs_test_t.max(axis=1)

        pred_df["student_pred_age_raw"] = pred_test_s
        pred_df["student_bag_raw"] = pred_test_s - y_test
        pred_df["student_maxprob"] = probs_test_s.max(axis=1)

        pred_df["baseline_pred_age_raw"] = pred_test_bl
        pred_df["baseline_bag_raw"] = pred_test_bl - y_test
        pred_df["baseline_maxprob"] = probs_test_bl.max(axis=1)
        pred_df.to_csv(os.path.join(fold_dir, "hc_dev_test_predictions_raw.csv"), index=False)
        dev_oof_rows.append(pred_df)

        train_cos_mean = float(np.nanmean(s_train_result.history["train_grad_cosine_sup_vs_kd"]))
        val_cos_mean = float(np.nanmean(s_train_result.history["val_grad_cosine_sup_vs_kd"]))
        best_idx = s_train_result.best_epoch - 1
        train_cos_best = float(s_train_result.history["train_grad_cosine_sup_vs_kd"][best_idx]) if 0 <= best_idx < len(s_train_result.history["train_grad_cosine_sup_vs_kd"]) else np.nan
        val_cos_best = float(s_train_result.history["val_grad_cosine_sup_vs_kd"][best_idx]) if 0 <= best_idx < len(s_train_result.history["val_grad_cosine_sup_vs_kd"]) else np.nan

        fold_rows.append({
            "fold": fold,
            "n_train_hc_dev": len(df_train),
            "n_val_hc_dev": len(df_val),
            "n_test_hc_dev": len(df_test),
            "n_classes": n_classes,
            "teacher_best_epoch": t_train_result.best_epoch,
            "student_best_epoch": s_train_result.best_epoch,
            "baseline_best_epoch": bl_train_result.best_epoch,
            "teacher_mae_test_hc_dev": t_mae,
            "teacher_rmse_test_hc_dev": t_rmse,
            "teacher_r2_test_hc_dev": t_r2,
            "teacher_r_test_hc_dev": t_r,
            "student_mae_test_hc_dev": s_mae,
            "student_rmse_test_hc_dev": s_rmse,
            "student_r2_test_hc_dev": s_r2,
            "student_r_test_hc_dev": s_r,
            "student_train_grad_cosine_sup_vs_kd_mean": train_cos_mean,
            "student_val_grad_cosine_sup_vs_kd_mean": val_cos_mean,
            "student_train_grad_cosine_sup_vs_kd_best_epoch": train_cos_best,
            "student_val_grad_cosine_sup_vs_kd_best_epoch": val_cos_best,
            "baseline_mae_test_hc_dev": bl_mae,
            "baseline_rmse_test_hc_dev": bl_rmse,
            "baseline_r2_test_hc_dev": bl_r2,
            "baseline_r_test_hc_dev": bl_r,
        })

    dev_summary = pd.DataFrame(fold_rows)
    dev_summary.to_csv(os.path.join(args.out_dir, "hc_dev_cv_summary.csv"), index=False)

    if not dev_oof_rows:
        raise RuntimeError("No HC-development OOF rows were produced.")

    dev_oof_df = pd.concat(dev_oof_rows, axis=0, ignore_index=True)
    dev_oof_df = dev_oof_df.sort_values([args.eid_col]).reset_index(drop=True)
    dev_oof_df.to_csv(os.path.join(args.out_dir, "hc_dev_cv_raw_predictions.csv"), index=False)

    # ------------------------------------------------------------------
    # Stage A2: fixed epoch counts from HC-development CV
    # ------------------------------------------------------------------
    final_teacher_epochs = median_int(teacher_best_epochs)
    final_student_epochs = median_int(student_best_epochs)
    final_baseline_epochs = median_int(baseline_best_epochs)

    print("\n" + "=" * 90)
    print("Final training settings selected from HC-development CV:")
    print(f"Teacher final epochs  = {final_teacher_epochs}")
    print(f"Student final epochs  = {final_student_epochs}")
    print(f"Baseline final epochs = {final_baseline_epochs}")

    # ------------------------------------------------------------------
    # Stage B: fixed-settings cross-fitting on ALL HC development
    # ------------------------------------------------------------------
    dev_age_grid_t = make_age_grid(
        torch.from_numpy(hc_dev_df[args.age_col].to_numpy().astype(np.float32)),
        age_bin_step=args.age_bin_step,
        margin_bins=args.age_grid_margin_bins,
    )
    dev_age_grid_np = dev_age_grid_t.cpu().numpy().astype(np.float32)

    Xs_dev_raw = hc_dev_df[smri_feat_cols].to_numpy().astype(np.float32)
    Xr_dev_raw = hc_dev_df[reg_feat_cols].to_numpy()
    y_dev = hc_dev_df[args.age_col].to_numpy().astype(np.float32)
    conf_dev = conf_frame(hc_dev_df)

    strata_all_dev = make_strata(
        hc_dev_df,
        age_col=args.age_col,
        n_age_bins=args.age_bins,
        stratify_site=args.stratify_site,
        site_prefix=args.site_prefix,
        stratify_sex=args.stratify_sex,
        sex_col=args.sex_col,
    )

    teacher_logits_dev_oof, teacher_pred_dev_oof = teacher_crossfit_logits_and_pred_fixed(
        X_raw=Xs_dev_raw,
        y=y_dev,
        conf_all=conf_dev,
        teacher_modality=teacher_modality,
        teacher_feat_cols=smri_feat_cols,
        site_cols=site_cols_conf,
        age_grid_np=dev_age_grid_np,
        label_sigma_t=label_sigma_t,
        teacher_cfg=teacher_cfg,
        device=device,
        crossfit_folds=args.k_folds,
        crossfit_seed=args.seed + 5000,
        batch_size=args.batch_size,
        lr=args.teacher_lr,
        weight_decay=teacher_weight_decay,
        fixed_epochs=final_teacher_epochs,
        strata=strata_all_dev,
    )

    student_pred_dev_oof = student_crossfit_pred_fixed(
        X_raw=Xr_dev_raw,
        y=y_dev,
        teacher_feature_raw=Xs_dev_raw,
        conf_all=conf_dev,
        teacher_modality=teacher_modality,
        teacher_feat_cols=smri_feat_cols,
        student_modality=student_modality,
        student_feat_cols=reg_feat_cols,
        site_cols=site_cols_conf,
        age_grid_np=dev_age_grid_np,
        label_sigma_t=label_sigma_t,
        label_sigma_s=label_sigma_s,
        teacher_cfg=teacher_cfg,
        student_cfg=student_cfg,
        device=device,
        crossfit_folds=args.k_folds,
        crossfit_seed=args.seed + 7000,
        inner_folds=args.inner_folds,
        batch_size=args.batch_size,
        teacher_lr=args.teacher_lr,
        student_lr=args.student_lr,
        teacher_weight_decay=teacher_weight_decay,
        student_weight_decay=student_weight_decay,
        teacher_fixed_epochs=final_teacher_epochs,
        student_fixed_epochs=final_student_epochs,
        lambda_kd=float(args.lambda_kd),
        rkd_scale=float(args.rkd_scale),
        use_residual_bag=use_residual_bag,
        kd_method=args.kd_method,
        distill_temperature=args.distill_temperature,
        strata=strata_all_dev,
    )

    baseline_pred_dev_oof = baseline_crossfit_pred_fixed(
        X_raw=Xr_dev_raw,
        y=y_dev,
        conf_all=conf_dev,
        student_modality=student_modality,
        student_feat_cols=reg_feat_cols,
        site_cols=site_cols_conf,
        age_grid_np=dev_age_grid_np,
        label_sigma_s=label_sigma_s,
        baseline_cfg=baseline_cfg,
        device=device,
        crossfit_folds=args.k_folds,
        crossfit_seed=args.seed + 7000,
        batch_size=args.batch_size,
        baseline_lr=args.baseline_lr,
        weight_decay=baseline_weight_decay,
        baseline_fixed_epochs=final_baseline_epochs,
        strata=strata_all_dev,
    )

    a_t, b_t = fit_bag_bias_correction(y_dev, teacher_pred_dev_oof)
    a_s, b_s = fit_bag_bias_correction(y_dev, student_pred_dev_oof)
    a_bl, b_bl = fit_bag_bias_correction(y_dev, baseline_pred_dev_oof)

    dev_crossfit_df = hc_dev_df[tgt_keep_cols].copy()
    dev_crossfit_df["set_name"] = "hc_dev_crossfit_oof"
    dev_crossfit_df["kd_method"] = kd_method_tag
    dev_crossfit_df["teacher_pred_age_raw"] = teacher_pred_dev_oof
    dev_crossfit_df["teacher_pred_age_biascorr"], dev_crossfit_df["teacher_bag_biascorr"] = apply_bag_bias_correction(y_dev, teacher_pred_dev_oof, a_t, b_t)
    dev_crossfit_df["teacher_bag_raw"] = teacher_pred_dev_oof - y_dev
    dev_crossfit_df["student_pred_age_raw"] = student_pred_dev_oof
    dev_crossfit_df["student_pred_age_biascorr"], dev_crossfit_df["student_bag_biascorr"] = apply_bag_bias_correction(y_dev, student_pred_dev_oof, a_s, b_s)
    dev_crossfit_df["student_bag_raw"] = student_pred_dev_oof - y_dev
    dev_crossfit_df["baseline_pred_age_raw"] = baseline_pred_dev_oof
    dev_crossfit_df["baseline_pred_age_biascorr"], dev_crossfit_df["baseline_bag_biascorr"] = apply_bag_bias_correction(y_dev, baseline_pred_dev_oof, a_bl, b_bl)
    dev_crossfit_df["baseline_bag_raw"] = baseline_pred_dev_oof - y_dev
    dev_crossfit_df.to_csv(os.path.join(args.out_dir, "hc_dev_crossfit_oof_predictions_biascorr.csv"), index=False)

    print("BAG bias correction estimated from HC-development cross-fit OOF predictions only.")

    # ------------------------------------------------------------------
    # Stage C1: final training on all HC development
    # ------------------------------------------------------------------
    set_seed(args.seed + 400_000)
    final_teacher, final_teacher_dec, final_teacher_hist = train_final_teacher_on_all_dev(
        Xs_dev_raw,
        y_dev,
        conf_all=conf_dev,
        teacher_modality=teacher_modality,
        teacher_feat_cols=smri_feat_cols,
        site_cols=site_cols_conf,
        age_grid_np=dev_age_grid_np,
        label_sigma_t=label_sigma_t,
        teacher_cfg=teacher_cfg,
        device=device,
        batch_size=args.batch_size,
        lr=args.teacher_lr,
        weight_decay=teacher_weight_decay,
        n_epochs=final_teacher_epochs,
    )
    set_seed(args.seed + 600_000)
    final_student, final_student_dec, final_student_hist = train_final_student_on_all_dev(
        Xr_dev_raw,
        y_dev,
        teacher_logits_dev_oof,
        conf_all=conf_dev,
        student_modality=student_modality,
        student_feat_cols=reg_feat_cols,
        site_cols=site_cols_conf,
        age_grid_np=dev_age_grid_np,
        label_sigma_s=label_sigma_s,
        student_cfg=student_cfg,
        device=device,
        batch_size=args.batch_size,
        lr=args.student_lr,
        weight_decay=student_weight_decay,
        n_epochs=final_student_epochs,
        lambda_kd=float(args.lambda_kd),
        rkd_scale=float(args.rkd_scale),
        use_residual_bag=use_residual_bag,
        kd_method=args.kd_method,
        distill_temperature=args.distill_temperature,
    )
    set_seed(args.seed + 600_000)
    final_baseline, final_baseline_dec, final_baseline_hist = train_final_baseline_on_all_dev(
        Xr_dev_raw,
        y_dev,
        conf_all=conf_dev,
        student_modality=student_modality,
        student_feat_cols=reg_feat_cols,
        site_cols=site_cols_conf,
        age_grid_np=dev_age_grid_np,
        label_sigma_s=label_sigma_s,
        baseline_cfg=baseline_cfg,
        device=device,
        batch_size=args.batch_size,
        lr=args.baseline_lr,
        weight_decay=baseline_weight_decay,
        n_epochs=final_baseline_epochs,
    )

    pd.DataFrame(final_teacher_hist).to_csv(os.path.join(args.out_dir, "final_teacher_history.csv"), index=False)
    pd.DataFrame(final_student_hist).to_csv(os.path.join(args.out_dir, "final_student_gd_history.csv"), index=False)
    pd.DataFrame(final_baseline_hist).to_csv(os.path.join(args.out_dir, "final_baseline_history.csv"), index=False)

    torch.save(final_teacher.state_dict(), os.path.join(args.out_dir, "final_teacher_state.pt"))
    torch.save(final_student.state_dict(), os.path.join(args.out_dir, "final_student_state.pt"))
    torch.save(final_baseline.state_dict(), os.path.join(args.out_dir, "final_baseline_state.pt"))

    # ------------------------------------------------------------------
    # Stage C2: final predictions on untouched HC normative holdout
    # ------------------------------------------------------------------
    Xs_norm_raw = hc_norm_df[smri_feat_cols].to_numpy().astype(np.float32)
    Xr_norm_raw = hc_norm_df[reg_feat_cols].to_numpy()
    y_norm = hc_norm_df[args.age_col].to_numpy().astype(np.float32)

    conf_norm = conf_frame(hc_norm_df)
    Xs_norm = final_teacher_dec.transform(Xs_norm_raw, conf_norm)
    Xr_norm_s = final_student_dec.transform(Xr_norm_raw, conf_norm)
    Xr_norm_bl = final_baseline_dec.transform(Xr_norm_raw, conf_norm)

    norm_eval_teacher = make_eval_loader(Xs_norm, y_norm, args.batch_size, False)
    norm_eval_student = make_eval_loader(Xr_norm_s, y_norm, args.batch_size, False)
    norm_eval_baseline = make_eval_loader(Xr_norm_bl, y_norm, args.batch_size, False)

    _, pred_norm_t, _, probs_norm_t = predict_teacher(final_teacher, norm_eval_teacher, age_grid=dev_age_grid_t, device=device)
    _, pred_norm_s, _, probs_norm_s = predict_student(final_student, norm_eval_student, age_grid=dev_age_grid_t, device=device)
    _, pred_norm_bl, _, probs_norm_bl = predict_student(final_baseline, norm_eval_baseline, age_grid=dev_age_grid_t, device=device)

    norm_pred_df = hc_norm_df[tgt_keep_cols].copy()
    norm_pred_df["set_name"] = "hc_norm_holdout"
    norm_pred_df["trained_on_hc_dev_only"] = 1
    norm_pred_df["age_grid_min"] = float(dev_age_grid_np[0])
    norm_pred_df["age_grid_max"] = float(dev_age_grid_np[-1])
    norm_pred_df["age_grid_step"] = float(args.age_bin_step)
    norm_pred_df["label_sigma_t"] = float(label_sigma_t)
    norm_pred_df["label_sigma_s"] = float(label_sigma_s)
    norm_pred_df["kd_method"] = kd_method_tag
    norm_pred_df["rkd_scale"] = float(args.rkd_scale)
    norm_pred_df["rkd_target"] = args.rkd_target
    norm_pred_df["lambda_kd"] = float(args.lambda_kd)

    norm_pred_df["teacher_pred_age_raw"] = pred_norm_t
    norm_pred_df["teacher_pred_age_biascorr"], norm_pred_df["teacher_bag_biascorr"] = apply_bag_bias_correction(y_norm, pred_norm_t, a_t, b_t)
    norm_pred_df["teacher_bag_raw"] = pred_norm_t - y_norm
    norm_pred_df["teacher_maxprob"] = probs_norm_t.max(axis=1)

    norm_pred_df["student_pred_age_raw"] = pred_norm_s
    norm_pred_df["student_pred_age_biascorr"], norm_pred_df["student_bag_biascorr"] = apply_bag_bias_correction(y_norm, pred_norm_s, a_s, b_s)
    norm_pred_df["student_bag_raw"] = pred_norm_s - y_norm
    norm_pred_df["student_maxprob"] = probs_norm_s.max(axis=1)

    norm_pred_df["baseline_pred_age_raw"] = pred_norm_bl
    norm_pred_df["baseline_pred_age_biascorr"], norm_pred_df["baseline_bag_biascorr"] = apply_bag_bias_correction(y_norm, pred_norm_bl, a_bl, b_bl)
    norm_pred_df["baseline_bag_raw"] = pred_norm_bl - y_norm
    norm_pred_df["baseline_maxprob"] = probs_norm_bl.max(axis=1)
    norm_pred_df.to_csv(os.path.join(args.out_dir, "hc_norm_holdout_predictions.csv"), index=False)

    # ------------------------------------------------------------------
    # Stage C3: final predictions on all cases
    # ------------------------------------------------------------------
    if len(case_df) > 0:
        Xs_case_raw = case_df[smri_feat_cols].to_numpy().astype(np.float32)
        Xr_case_raw = case_df[reg_feat_cols].to_numpy()
        y_case = case_df[args.age_col].to_numpy().astype(np.float32)

        conf_case = conf_frame(case_df)
        Xs_case = final_teacher_dec.transform(Xs_case_raw, conf_case)
        Xr_case_s = final_student_dec.transform(Xr_case_raw, conf_case)
        Xr_case_bl = final_baseline_dec.transform(Xr_case_raw, conf_case)

        case_eval_teacher = make_eval_loader(Xs_case, y_case, args.batch_size, False)
        case_eval_student = make_eval_loader(Xr_case_s, y_case, args.batch_size, False)
        case_eval_baseline = make_eval_loader(Xr_case_bl, y_case, args.batch_size, False)

        _, pred_case_t, _, probs_case_t = predict_teacher(final_teacher, case_eval_teacher, age_grid=dev_age_grid_t, device=device)
        _, pred_case_s, _, probs_case_s = predict_student(final_student, case_eval_student, age_grid=dev_age_grid_t, device=device)
        _, pred_case_bl, _, probs_case_bl = predict_student(final_baseline, case_eval_baseline, age_grid=dev_age_grid_t, device=device)

        case_pred_df = case_df[tgt_keep_cols].copy()
        case_pred_df["set_name"] = "cases"
        case_pred_df["trained_on_hc_dev_only"] = 1
        case_pred_df["age_grid_min"] = float(dev_age_grid_np[0])
        case_pred_df["age_grid_max"] = float(dev_age_grid_np[-1])
        case_pred_df["age_grid_step"] = float(args.age_bin_step)
        case_pred_df["label_sigma_t"] = float(label_sigma_t)
        case_pred_df["label_sigma_s"] = float(label_sigma_s)
        case_pred_df["kd_method"] = kd_method_tag
        case_pred_df["rkd_scale"] = float(args.rkd_scale)
        case_pred_df["rkd_target"] = args.rkd_target
        case_pred_df["lambda_kd"] = float(args.lambda_kd)

        case_pred_df["teacher_pred_age_raw"] = pred_case_t
        case_pred_df["teacher_pred_age_biascorr"], case_pred_df["teacher_bag_biascorr"] = apply_bag_bias_correction(y_case, pred_case_t, a_t, b_t)
        case_pred_df["teacher_bag_raw"] = pred_case_t - y_case
        case_pred_df["teacher_maxprob"] = probs_case_t.max(axis=1)

        case_pred_df["student_pred_age_raw"] = pred_case_s
        case_pred_df["student_pred_age_biascorr"], case_pred_df["student_bag_biascorr"] = apply_bag_bias_correction(y_case, pred_case_s, a_s, b_s)
        case_pred_df["student_bag_raw"] = pred_case_s - y_case
        case_pred_df["student_maxprob"] = probs_case_s.max(axis=1)

        case_pred_df["baseline_pred_age_raw"] = pred_case_bl
        case_pred_df["baseline_pred_age_biascorr"], case_pred_df["baseline_bag_biascorr"] = apply_bag_bias_correction(y_case, pred_case_bl, a_bl, b_bl)
        case_pred_df["baseline_bag_raw"] = pred_case_bl - y_case
        case_pred_df["baseline_maxprob"] = probs_case_bl.max(axis=1)
        case_pred_df.to_csv(os.path.join(args.out_dir, "cases_final_predictions.csv"), index=False)
        print(f"Wrote final case predictions for N={len(case_df)} cases.")
    else:
        print("No cases available after filtering; skipped final case scoring.")

    # ------------------------------------------------------------------
    # Metadata and summary
    # ------------------------------------------------------------------
    final_meta = {
        "kd_method": kd_method_tag,
        "rkd_target": args.rkd_target,
        "rkd_scale": float(args.rkd_scale),
        "lambda_kd": float(args.lambda_kd),
        "label_sigma_t": float(label_sigma_t),
        "label_sigma_s": float(label_sigma_s),
        "norm_holdout_frac": float(args.norm_holdout_frac),
        "k_folds_hc_dev_cv": int(args.k_folds),
        "k_folds_hc_dev_crossfit": int(args.k_folds),
        "inner_folds_for_teacher_oof_train_targets": int(args.inner_folds),
        "n_hc_dev": int(len(hc_dev_df)),
        "n_hc_norm_holdout": int(len(hc_norm_df)),
        "n_cases": int(len(case_df)),
        "final_teacher_epochs": int(final_teacher_epochs),
        "final_student_epochs": int(final_student_epochs),
        "final_baseline_epochs": int(final_baseline_epochs),
        "teacher_bag_bias_a": float(a_t),
        "teacher_bag_bias_b": float(b_t),
        "student_bag_bias_a": float(a_s),
        "student_bag_bias_b": float(b_s),
        "baseline_bag_bias_a": float(a_bl),
        "baseline_bag_bias_b": float(b_bl),
        "distill_temperature": float(args.distill_temperature),
        "teacher_rep_hidden": args.teacher_rep_hidden,
        "teacher_rep_dim": int(args.teacher_rep_dim),
        "student_rep_hidden": args.student_rep_hidden,
        "student_rep_dim": int(args.student_rep_dim),
        "teacher_lr": float(args.teacher_lr),
        "student_lr": float(args.student_lr),
        "baseline_lr": float(args.baseline_lr),
        "teacher_dropout": float(teacher_dropout),
        "student_dropout": float(student_dropout),
        "baseline_dropout": float(baseline_dropout),
        "teacher_weight_decay": float(teacher_weight_decay),
        "student_weight_decay": float(student_weight_decay),
        "baseline_weight_decay": float(baseline_weight_decay),
    }
    with open(os.path.join(args.out_dir, "final_training_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(final_meta, f, indent=2)

    def mean_std(x: np.ndarray) -> Tuple[float, float]:
        return float(np.nanmean(x)), float(np.nanstd(x))

    print("\n" + "=" * 90)
    print("HC-development cross-validated evaluation summary:")
    for prefix, label in [("teacher", "Teacher"), ("student", "StudentGD"), ("baseline", "Baseline")]:
        m_mae, s_mae = mean_std(dev_summary[f"{prefix}_mae_test_hc_dev"].to_numpy())
        m_rmse, s_rmse = mean_std(dev_summary[f"{prefix}_rmse_test_hc_dev"].to_numpy())
        m_r2, s_r2 = mean_std(dev_summary[f"{prefix}_r2_test_hc_dev"].to_numpy())
        m_r, s_r = mean_std(dev_summary[f"{prefix}_r_test_hc_dev"].to_numpy())
        print(
            f"{label:9s} | MAE {m_mae:.3f} ± {s_mae:.3f} | RMSE {m_rmse:.3f} ± {s_rmse:.3f} | "
            f"R2 {m_r2:.3f} ± {s_r2:.3f} | r {m_r:.3f} ± {s_r:.3f}"
        )

    student_cos_train_mean, student_cos_train_sd = mean_std(dev_summary["student_train_grad_cosine_sup_vs_kd_mean"].to_numpy())
    student_cos_val_mean, student_cos_val_sd = mean_std(dev_summary["student_val_grad_cosine_sup_vs_kd_mean"].to_numpy())
    print(
        f"Student gradient cosine similarity across HC-development folds | "
        f"train mean {student_cos_train_mean:.4f} ± {student_cos_train_sd:.4f} | "
        f"val mean {student_cos_val_mean:.4f} ± {student_cos_val_sd:.4f}"
    )

    print("\nHC normative holdout BAG summary (bias-corrected):")
    for prefix, label in [("teacher", "Teacher"), ("student", "StudentGD"), ("baseline", "Baseline")]:
        m, s = bag_stats(norm_pred_df[f"{prefix}_bag_biascorr"].to_numpy().astype(np.float32))
        print(f"{label:9s} | mean {m:+.3f} | std {s:.3f}")

    if len(case_df) > 0:
        print("\nCases BAG summary (bias-corrected):")
        for prefix, label in [("teacher", "Teacher"), ("student", "StudentGD"), ("baseline", "Baseline")]:
            m, s = bag_stats(case_pred_df[f"{prefix}_bag_biascorr"].to_numpy().astype(np.float32))
            print(f"{label:9s} | mean {m:+.3f} | std {s:.3f}")

    print(f"\nWrote outputs to: {args.out_dir}")

if __name__ == "__main__":
    main()