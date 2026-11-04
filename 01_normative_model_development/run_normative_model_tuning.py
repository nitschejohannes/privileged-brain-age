#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from teacher_and_student_models import GDModelConfig, StudentClassifier, TeacherClassifier, make_age_grid
from baseline import make_baseline_model
from pipeline_deconfound import (
    FoldDeconfounder,
    resolve_confound_cols,
    required_confounds_for,
    split_features_and_qc,
)

from run_normative_modeling_pipeline import (
    set_seed,
    ensure_dir,
    parse_cols,
    parse_int_tuple,
    cols_by_prefix,
    uniq,
    assert_unique_eids,
    compute_metrics,
    make_strata,
    make_soft_targets_np,
    make_soft_loader,
    make_eval_loader,
    make_kd_loader,
    _safe_drop_last,
    train_supervised_classifier,
    train_student_distilled,
    predict_student,
    predict_teacher,
    teacher_oof_logits_on_train_with_early_stopping,
)

METRIC_COLS = ["mae", "rmse", "r2", "r"]
LOWER_IS_BETTER = {"mae", "rmse"}

# -----------------------------------------------------------------------------
# Fixed context shared by every tuning run
# -----------------------------------------------------------------------------
@dataclass
class Ctx:
    age_col: str
    sex_col: str
    site_prefix: str
    teacher_feat_cols: List[str]
    student_feat_cols: List[str]
    teacher_modality: str
    student_modality: str
    site_cols: List[str]
    conf_cols_all: List[str]
    activation: str
    layer_norm: bool
    teacher_rep_hidden: Tuple[int, ...]
    teacher_rep_dim: int
    teacher_head_hidden: Tuple[int, ...]
    student_rep_hidden: Tuple[int, ...]
    student_rep_dim: int
    student_head_hidden: Tuple[int, ...]
    device: torch.device
    k_folds: int
    inner_folds: int
    val_frac: float
    age_bins: int
    stratify_site: bool
    stratify_sex: bool
    age_bin_step: float
    age_grid_margin_bins: int
    label_sigma_s: float
    label_sigma_t: float
    batch_size: int
    max_epochs: int
    patience: int
    # KD settings (used only by the distilled student stage at lambda > 0)
    rkd_scale: float
    use_residual_bag: bool
    kd_method: str
    distill_temperature: float

    def teacher_cfg(self, dropout: float) -> GDModelConfig:
        return GDModelConfig(
            activation=self.activation, dropout=dropout, layer_norm=self.layer_norm,
            rep_hidden=self.teacher_rep_hidden, rep_dim=self.teacher_rep_dim,
            head_hidden=self.teacher_head_hidden,
        )

    def student_cfg(self, dropout: float) -> GDModelConfig:
        return GDModelConfig(
            activation=self.activation, dropout=dropout, layer_norm=self.layer_norm,
            rep_hidden=self.student_rep_hidden, rep_dim=self.student_rep_dim,
            head_hidden=self.student_head_hidden,
        )

    def conf_frame(self, sub_df: pd.DataFrame) -> pd.DataFrame:
        if self.conf_cols_all:
            return sub_df[self.conf_cols_all].reset_index(drop=True)
        return pd.DataFrame(index=range(len(sub_df))).reset_index(drop=True)

    def strata(self, df: pd.DataFrame) -> np.ndarray:
        return make_strata(
            df, age_col=self.age_col, n_age_bins=self.age_bins,
            stratify_site=self.stratify_site, site_prefix=self.site_prefix,
            stratify_sex=self.stratify_sex, sex_col=self.sex_col,
        )

# -----------------------------------------------------------------------------
# Data preparation (mirrors pipeline.py up to the HC-dev / holdout split)
# -----------------------------------------------------------------------------
def prepare(args: argparse.Namespace) -> Tuple[Ctx, pd.DataFrame]:
    device = torch.device(
        args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    )
    print(f"Using device: {device}")

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

    teacher_excl_cols = [c for c in parse_cols(args.teacher_cov_cols) if c in df.columns]
    teacher_excl_cols += cols_by_prefix(df, parse_cols(args.teacher_cov_prefixes))
    teacher_excl_cols = uniq(teacher_excl_cols)
    student_excl_cols = [c for c in parse_cols(args.student_cov_cols) if c in df.columns]
    student_excl_cols += cols_by_prefix(df, parse_cols(args.student_cov_prefixes))
    student_excl_cols = uniq(student_excl_cols)

    smri_numeric = [c for c in smri.columns
                    if c != args.eid_col and c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    reg_numeric = [c for c in reg.columns
                   if c != args.eid_col and c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    smri_feat_cols = [c for c in smri_numeric if c not in set(teacher_excl_cols)]
    reg_feat_cols = [c for c in reg_numeric if c not in set(student_excl_cols)]

    smri_feat_cols, smri_qc_cols = split_features_and_qc(smri_feat_cols)
    reg_feat_cols, reg_qc_cols = split_features_and_qc(reg_feat_cols)
    if smri_qc_cols or reg_qc_cols:
        print(f"Quarantined QC columns | teacher-block: {smri_qc_cols} | student-block: {reg_qc_cols}")
    if len(smri_feat_cols) == 0 or len(reg_feat_cols) == 0:
        raise ValueError("Could not infer teacher/student feature columns after exclusions.")

    conf_info = resolve_confound_cols(df.columns, site_prefix=args.site_prefix)
    site_cols_conf = conf_info["site_cols"]
    teacher_modality = args.teacher_modality
    student_modality = args.student_modality
    if args.deconfound:
        need_conf = uniq(required_confounds_for(teacher_modality, site_cols_conf)
                         + required_confounds_for(student_modality, site_cols_conf))
        missing_conf = [c for c in need_conf if c not in df.columns]
        if missing_conf:
            raise ValueError(f"--deconfound requested but confound columns missing: {missing_conf}. "
                             f"Present site dummies: {site_cols_conf}.")
        print(f"Deconfounding ON | teacher='{teacher_modality}' student='{student_modality}' | confounds: {need_conf}")
    else:
        teacher_modality = "questionnaire"
        student_modality = "questionnaire"
        print("Deconfounding OFF | z-score only")

    conf_cols_all = uniq(required_confounds_for(args.teacher_modality, site_cols_conf)
                         + required_confounds_for(args.student_modality, site_cols_conf))
    conf_cols_all = [c for c in conf_cols_all if c in df.columns]

    needed = [args.age_col, args.hc_col] + teacher_excl_cols + student_excl_cols + smri_feat_cols + reg_feat_cols
    needed = uniq([c for c in needed if c in df.columns])
    df = df.dropna(subset=needed).copy()
    assert_unique_eids(df, name="post-dropna merged dataframe", eid_col=args.eid_col)
    print(f"After dropping NaNs: N={len(df)}")

    hc_df = df[df[args.hc_col] == args.hc_value].copy().reset_index(drop=True)
    if len(hc_df) == 0:
        raise ValueError("No healthy controls available after filtering.")
    print(f"HC N={len(hc_df)}")
    print(f"Teacher-block features: {len(smri_feat_cols)} | student-block features: {len(reg_feat_cols)}")

    ctx = Ctx(
        age_col=args.age_col, sex_col=args.sex_col, site_prefix=args.site_prefix,
        teacher_feat_cols=smri_feat_cols, student_feat_cols=reg_feat_cols,
        teacher_modality=teacher_modality, student_modality=student_modality,
        site_cols=site_cols_conf, conf_cols_all=conf_cols_all,
        activation=args.activation, layer_norm=(not args.no_layer_norm),
        teacher_rep_hidden=parse_int_tuple(args.teacher_rep_hidden), teacher_rep_dim=args.teacher_rep_dim,
        teacher_head_hidden=parse_int_tuple(args.teacher_head_hidden),
        student_rep_hidden=parse_int_tuple(args.student_rep_hidden), student_rep_dim=args.student_rep_dim,
        student_head_hidden=parse_int_tuple(args.student_head_hidden),
        device=device, k_folds=args.k_folds, inner_folds=args.inner_folds, val_frac=args.val_frac,
        age_bins=args.age_bins, stratify_site=args.stratify_site, stratify_sex=args.stratify_sex,
        age_bin_step=args.age_bin_step, age_grid_margin_bins=args.age_grid_margin_bins,
        label_sigma_s=args.label_sigma_s, label_sigma_t=args.label_sigma_t,
        batch_size=args.batch_size, max_epochs=args.max_epochs, patience=args.patience,
        rkd_scale=args.rkd_scale, use_residual_bag=(args.rkd_target == "bag"),
        kd_method=args.kd_method, distill_temperature=args.distill_temperature,
    )

    hc_strata_full = ctx.strata(hc_df)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.norm_holdout_frac, random_state=args.split_seed)
    dev_idx, norm_idx = next(sss.split(np.zeros(len(hc_df)), hc_strata_full))
    hc_dev_df = hc_df.iloc[dev_idx].copy().reset_index(drop=True)
    print(f"HC-dev N={len(hc_dev_df)} | (holdout N={len(norm_idx)} reserved, NOT used in tuning)")
    print(f"Split seed={args.split_seed} (matches pipeline.py --seed {args.split_seed - 17})")
    return ctx, hc_dev_df

# -----------------------------------------------------------------------------
# Fold iterator (shared by all stages -> paired comparisons per seed)
# -----------------------------------------------------------------------------
def _fold_split(df_dev: pd.DataFrame, ctx: Ctx, seed: int):
    strata_dev = ctx.strata(df_dev)
    skf = StratifiedKFold(n_splits=ctx.k_folds, shuffle=True, random_state=seed)
    for fold, (tv_idx, te_idx) in enumerate(skf.split(df_dev, strata_dev), start=1):
        df_tv = df_dev.iloc[tv_idx].copy().reset_index(drop=True)
        df_te = df_dev.iloc[te_idx].copy().reset_index(drop=True)
        grid_t = make_age_grid(
            torch.from_numpy(df_tv[ctx.age_col].to_numpy().astype(np.float32)),
            age_bin_step=ctx.age_bin_step, margin_bins=ctx.age_grid_margin_bins,
        )
        grid_np = grid_t.cpu().numpy().astype(np.float32)
        strata_tv = ctx.strata(df_tv)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=ctx.val_frac, random_state=seed + fold)
        tr_rel, va_rel = next(sss.split(np.zeros(len(df_tv)), strata_tv))
        df_tr = df_tv.iloc[tr_rel].copy().reset_index(drop=True)
        df_va = df_tv.iloc[va_rel].copy().reset_index(drop=True)
        yield fold, te_idx, df_tr, df_va, df_te, grid_t, grid_np

def metrics_from_oof(y_true: np.ndarray, oof: np.ndarray) -> Dict[str, float]:
    mae, rmse, r2, r = compute_metrics(y_true, oof)
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2), "r": float(r)}

# -----------------------------------------------------------------------------
# Supervised CV OOF  (teacher, or baseline == student @ lambda 0)
# -----------------------------------------------------------------------------
def supervised_cv_oof(
    df_dev: pd.DataFrame, ctx: Ctx, *,
    feat_cols: List[str], modality: str, cfg: GDModelConfig, sigma: float,
    is_teacher: bool, lr: float, weight_decay: float, seed: int,
) -> Tuple[Dict[str, float], List[int]]:
    oof = np.full(len(df_dev), np.nan, dtype=np.float32)
    y_true = df_dev[ctx.age_col].to_numpy().astype(np.float32)
    best_epochs: List[int] = []
    seed_offset = 100_000 if is_teacher else 300_000
    tag = "T" if is_teacher else "BL"

    for fold, te_idx, df_tr, df_va, df_te, grid_t, grid_np in _fold_split(df_dev, ctx, seed):
        n_classes = len(grid_np)
        X_tr_raw, X_va_raw, X_te_raw = (df_tr[feat_cols].to_numpy(),
                                        df_va[feat_cols].to_numpy(),
                                        df_te[feat_cols].to_numpy())
        y_tr = df_tr[ctx.age_col].to_numpy().astype(np.float32)
        y_va = df_va[ctx.age_col].to_numpy().astype(np.float32)
        y_te = df_te[ctx.age_col].to_numpy().astype(np.float32)
        conf_tr, conf_va, conf_te = ctx.conf_frame(df_tr), ctx.conf_frame(df_va), ctx.conf_frame(df_te)

        dec = FoldDeconfounder(modality, feat_cols, site_cols=ctx.site_cols).fit(X_tr_raw, conf_tr)
        X_tr, X_va, X_te = dec.transform(X_tr_raw, conf_tr), dec.transform(X_va_raw, conf_va), dec.transform(X_te_raw, conf_te)
        yprob_tr = make_soft_targets_np(y_tr, grid_np, sigma=sigma)
        yprob_va = make_soft_targets_np(y_va, grid_np, sigma=sigma)
        tr_loader = make_soft_loader(X_tr, y_tr, yprob_tr, ctx.batch_size, True, drop_last=_safe_drop_last(len(y_tr), ctx.batch_size))
        va_loader = make_soft_loader(X_va, y_va, yprob_va, ctx.batch_size, False)
        te_loader = make_eval_loader(X_te, y_te, ctx.batch_size, False)

        set_seed(seed + seed_offset + fold)
        model = (TeacherClassifier(x_dim=X_tr.shape[1], n_classes=n_classes, cfg=cfg) if is_teacher
                 else make_baseline_model(x_dim=X_tr.shape[1], n_classes=n_classes, cfg=cfg))
        res = train_supervised_classifier(
            model, tr_loader, va_loader, age_grid=grid_t, device=ctx.device,
            lr=lr, weight_decay=weight_decay, max_epochs=ctx.max_epochs, patience=ctx.patience,
            verbose=False, tag=f"{tag}-s{seed}-f{fold}",
        )
        best_epochs.append(res.best_epoch)
        _, pred_te, _, _ = predict_student(model, te_loader, age_grid=grid_t, device=ctx.device)
        oof[te_idx] = pred_te.astype(np.float32)

    return metrics_from_oof(y_true, oof), best_epochs

# -----------------------------------------------------------------------------
# Distilled student CV OOF  (lambda > 0; frozen teacher, inner-CV OOF logits)
# -----------------------------------------------------------------------------
def student_distilled_cv_oof(
    df_dev: pd.DataFrame, ctx: Ctx, *,
    lr: float, weight_decay: float, dropout: float, student_lambda: float,
    distill_temperature: float, teacher_hp: Optional[Dict[str, float]], seed: int,
) -> Tuple[Dict[str, float], List[int]]:
    # distill_temperature is passed through to train_student_distilled. Under
    # kd_method="rkd" it is INERT (combined_student_classification_loss ignores it
    # entirely — RKD matches a scalar-BAG distance geometry, which has no softmax
    # to sharpen), so RKD behaviour is identical regardless of its value. It only
    # takes effect under kd_method="soft_label".
    need_teacher = student_lambda > 0.0
    if need_teacher and teacher_hp is None:
        raise ValueError("student_lambda > 0 requires a frozen teacher config, but none was provided.")
    oof = np.full(len(df_dev), np.nan, dtype=np.float32)
    y_true = df_dev[ctx.age_col].to_numpy().astype(np.float32)
    best_epochs: List[int] = []
    student_cfg = ctx.student_cfg(dropout)
    teacher_cfg = ctx.teacher_cfg(teacher_hp["dropout"]) if need_teacher else None

    for fold, te_idx, df_tr, df_va, df_te, grid_t, grid_np in _fold_split(df_dev, ctx, seed):
        n_classes = len(grid_np)
        # ---- student-block features ----
        Xr_tr_raw, Xr_va_raw, Xr_te_raw = (df_tr[ctx.student_feat_cols].to_numpy(),
                                           df_va[ctx.student_feat_cols].to_numpy(),
                                           df_te[ctx.student_feat_cols].to_numpy())
        y_tr = df_tr[ctx.age_col].to_numpy().astype(np.float32)
        y_va = df_va[ctx.age_col].to_numpy().astype(np.float32)
        y_te = df_te[ctx.age_col].to_numpy().astype(np.float32)
        conf_tr, conf_va, conf_te = ctx.conf_frame(df_tr), ctx.conf_frame(df_va), ctx.conf_frame(df_te)

        s_dec = FoldDeconfounder(ctx.student_modality, ctx.student_feat_cols, site_cols=ctx.site_cols).fit(Xr_tr_raw, conf_tr)
        Xr_tr, Xr_va, Xr_te = s_dec.transform(Xr_tr_raw, conf_tr), s_dec.transform(Xr_va_raw, conf_va), s_dec.transform(Xr_te_raw, conf_te)
        yprob_tr_s = make_soft_targets_np(y_tr, grid_np, sigma=ctx.label_sigma_s)
        yprob_va_s = make_soft_targets_np(y_va, grid_np, sigma=ctx.label_sigma_s)

        # ---- frozen-config teacher (per fold), leakage-safe OOF logits ----
        if need_teacher:
            Xs_tr_raw = df_tr[ctx.teacher_feat_cols].to_numpy()
            Xs_va_raw = df_va[ctx.teacher_feat_cols].to_numpy()
            t_dec = FoldDeconfounder(ctx.teacher_modality, ctx.teacher_feat_cols, site_cols=ctx.site_cols).fit(Xs_tr_raw, conf_tr)
            Xs_tr, Xs_va = t_dec.transform(Xs_tr_raw, conf_tr), t_dec.transform(Xs_va_raw, conf_va)
            yprob_tr_t = make_soft_targets_np(y_tr, grid_np, sigma=ctx.label_sigma_t)
            yprob_va_t = make_soft_targets_np(y_va, grid_np, sigma=ctx.label_sigma_t)
            t_tr_loader = make_soft_loader(Xs_tr, y_tr, yprob_tr_t, ctx.batch_size, True)
            t_va_loader = make_soft_loader(Xs_va, y_va, yprob_va_t, ctx.batch_size, False)
            set_seed(seed + 100_000 + fold)
            teacher = TeacherClassifier(x_dim=Xs_tr.shape[1], n_classes=n_classes, cfg=teacher_cfg)
            train_supervised_classifier(
                teacher, t_tr_loader, t_va_loader, age_grid=grid_t, device=ctx.device,
                lr=teacher_hp["lr"], weight_decay=teacher_hp["weight_decay"],
                max_epochs=ctx.max_epochs, patience=ctx.patience, verbose=False, tag=f"T-s{seed}-f{fold}",
            )
            _, _, logits_va_t, _ = predict_teacher(
                teacher, make_eval_loader(Xs_va, y_va, ctx.batch_size, False),
                age_grid=grid_t, device=ctx.device,
            )
            strata_tr = ctx.strata(df_tr)
            teacher_logits_tr_oof = teacher_oof_logits_on_train_with_early_stopping(
                X_train_raw=Xs_tr_raw, y_train=y_tr, conf_train=conf_tr,
                teacher_modality=ctx.teacher_modality, teacher_feat_cols=ctx.teacher_feat_cols,
                site_cols=ctx.site_cols, age_grid_np=grid_np, label_sigma_t=ctx.label_sigma_t,
                teacher_cfg=teacher_cfg, device=ctx.device, inner_folds=ctx.inner_folds,
                inner_seed=seed + 1000 + fold, batch_size=ctx.batch_size,
                lr=teacher_hp["lr"], weight_decay=teacher_hp["weight_decay"],
                max_epochs=min(ctx.max_epochs, 120), patience=min(ctx.patience, 15),
                strata_train=strata_tr,
            )
        else:
            logits_va_t = np.zeros((len(y_va), n_classes), dtype=np.float32)
            teacher_logits_tr_oof = np.zeros((len(y_tr), n_classes), dtype=np.float32)

        te_loader = make_eval_loader(Xr_te, y_te, ctx.batch_size, False)
        s_drop_last = _safe_drop_last(len(y_tr), ctx.batch_size)
        s_tr_loader = make_kd_loader(Xr_tr, y_tr, yprob_tr_s, teacher_logits_tr_oof, ctx.batch_size, True, drop_last=s_drop_last)
        s_va_loader = make_kd_loader(Xr_va, y_va, yprob_va_s, logits_va_t, ctx.batch_size, False, drop_last=False)

        # Re-seed to the student's own init (independent of teacher RNG use), so
        # baseline and student share init per (seed, fold) -> paired comparison.
        set_seed(seed + 300_000 + fold)
        student = StudentClassifier(x_dim=Xr_tr.shape[1], n_classes=n_classes, cfg=student_cfg)
        res = train_student_distilled(
            student, s_tr_loader, s_va_loader, age_grid=grid_t, device=ctx.device,
            lr=lr, weight_decay=weight_decay, max_epochs=ctx.max_epochs, patience=ctx.patience,
            lambda_kd=student_lambda, rkd_scale=ctx.rkd_scale, use_residual_bag=ctx.use_residual_bag,
            kd_method=ctx.kd_method, distill_temperature=distill_temperature,
            verbose=False, tag=f"S-s{seed}-f{fold}-l{student_lambda:g}-T{distill_temperature:g}",
        )
        best_epochs.append(res.best_epoch)
        _, pred_te, _, _ = predict_student(student, te_loader, age_grid=grid_t, device=ctx.device)
        oof[te_idx] = pred_te.astype(np.float32)

    return metrics_from_oof(y_true, oof), best_epochs

# -----------------------------------------------------------------------------
# Sweep + aggregation
# -----------------------------------------------------------------------------
def summarize(df_rows: pd.DataFrame, group_cols: List[str], select_metric: str) -> pd.DataFrame:
    agg: Dict[str, Any] = {}
    for m in METRIC_COLS:
        agg[f"oof_{m}_mean"] = (f"oof_{m}", "mean")
        agg[f"oof_{m}_std"] = (f"oof_{m}", "std")
    agg["n_seeds"] = ("seed", "size")
    g = df_rows.groupby(group_cols, as_index=False).agg(**agg)
    for m in METRIC_COLS:
        g[f"oof_{m}_std"] = g[f"oof_{m}_std"].fillna(0.0)
    key = f"oof_{select_metric}_mean"
    ascending = select_metric in LOWER_IS_BETTER
    return g.sort_values(key, ascending=ascending).reset_index(drop=True)

def sweep(eval_fn: Callable[[float, float, float, float, int], Tuple[Dict[str, float], List[int]]],
          *, lrs, wds, drops, temps, seeds, select_metric: str, label: str,
          include_temp: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Grid over lr x weight_decay x dropout x (temperature) x seeds.

    include_temp selects joint selection over temperature: when True (soft-label
    KD at lambda>0 with more than one temperature) 'distill_temperature' becomes a
    real 4th grid axis and part of the group key, so the winner is the best
    (lr, wd, dropout, T) jointly. When False (teacher, baseline, RKD, or a single
    fixed temperature) temps has length 1, no temperature column is emitted, and
    the grouping/CSVs are byte-for-byte what they were before this feature.
    """
    rows: List[Dict[str, Any]] = []
    for lr in lrs:
        for wd in wds:
            for dr in drops:
                for temp in temps:
                    for seed in seeds:
                        metrics, epochs = eval_fn(lr, wd, dr, temp, seed)
                        row = {"lr": lr, "weight_decay": wd, "dropout": dr, "seed": seed,
                               "median_best_epoch": int(np.median(epochs))}
                        if include_temp:
                            row["distill_temperature"] = temp
                        for m in METRIC_COLS:
                            row[f"oof_{m}"] = metrics[m]
                        rows.append(row)
                        tstr = f" T={temp:g}" if include_temp else ""
                        print(f"  [{label}] lr={lr:g} wd={wd:g} dropout={dr:g}{tstr} seed={seed} "
                              f"-> MAE={metrics['mae']:.3f} RMSE={metrics['rmse']:.3f} "
                              f"R2={metrics['r2']:.3f} r={metrics['r']:.3f}")
    df = pd.DataFrame(rows)
    group_cols = ["lr", "weight_decay", "dropout"] + (["distill_temperature"] if include_temp else [])
    summ = summarize(df, group_cols, select_metric)
    return df, summ

def print_summary(summ: pd.DataFrame, select_metric: str, title: str) -> None:
    has_temp = "distill_temperature" in summ.columns
    print(f"\n{title} (sorted by mean OOF {select_metric.upper()}):")
    for _, row in summ.iterrows():
        marker = {"mae": row.oof_mae_mean, "rmse": row.oof_rmse_mean,
                  "r2": row.oof_r2_mean, "r": row.oof_r_mean}[select_metric]
        prefix = f"  lr={row.lr:g} wd={row.weight_decay:g} dropout={row.dropout:g}"
        if has_temp:
            prefix += f" T={row.distill_temperature:g}"
        print(f"{prefix} | "
              f"MAE {row.oof_mae_mean:.3f}±{row.oof_mae_std:.3f} | "
              f"RMSE {row.oof_rmse_mean:.3f}±{row.oof_rmse_std:.3f} | "
              f"R2 {row.oof_r2_mean:.3f}±{row.oof_r2_std:.3f} | "
              f"r {row.oof_r_mean:.3f}±{row.oof_r_std:.3f}"
              f"   <== select={marker:.4f}")

def config_from_row(row: pd.Series) -> Dict[str, float]:
    cfg = {
        "lr": float(row.lr), "weight_decay": float(row.weight_decay), "dropout": float(row.dropout),
        "oof_mae_mean": float(row.oof_mae_mean), "oof_mae_std": float(row.oof_mae_std),
        "oof_rmse_mean": float(row.oof_rmse_mean), "oof_r2_mean": float(row.oof_r2_mean),
        "oof_r_mean": float(row.oof_r_mean), "oof_r_std": float(row.oof_r_std),
    }
    if "distill_temperature" in row.index:
        cfg["distill_temperature"] = float(row.distill_temperature)
    return cfg

# -----------------------------------------------------------------------------
# Persisted-config helpers (option b: persist & reload the frozen teacher)
# -----------------------------------------------------------------------------
def load_configs_json(path: str) -> Dict[str, Any]:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def resolve_teacher_hp(args, ctx_in_memory: Optional[Dict[str, float]], out_dir: str) -> Optional[Dict[str, float]]:
    # 1) explicit CLI overrides (all three required)
    if args.teacher_lr is not None and args.teacher_weight_decay is not None and args.teacher_dropout is not None:
        return {"lr": float(args.teacher_lr), "weight_decay": float(args.teacher_weight_decay),
                "dropout": float(args.teacher_dropout), "source": "cli_override"}
    # 2) teacher stage run in this same invocation
    if ctx_in_memory is not None:
        return {**ctx_in_memory, "source": "in_memory_stage0"}
    # 3) persisted JSON (explicit path, else out_dir default)
    path = args.teacher_config_json or os.path.join(out_dir, "tune_best_configs.json")
    j = load_configs_json(path)
    if "teacher" in j and all(k in j["teacher"] for k in ("lr", "weight_decay", "dropout")):
        t = j["teacher"]
        return {"lr": float(t["lr"]), "weight_decay": float(t["weight_decay"]),
                "dropout": float(t["dropout"]), "source": f"json:{path}"}
    return None

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Independent teacher / baseline / student(lambda) HP tuning.")

    ap.add_argument("--smri_csv", type=str, required=True, help="Teacher-block CSV (put fMRI here if teacher=fMRI).")
    ap.add_argument("--reg_csv", type=str, required=True, help="Student-block CSV (put sMRI here if student=sMRI).")
    ap.add_argument("--target_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="tune_out")
    ap.add_argument("--eid_col", type=str, default="eid")
    ap.add_argument("--age_col", type=str, default="age_at_assessment")
    ap.add_argument("--hc_col", type=str, required=True)
    ap.add_argument("--hc_value", type=int, default=1)
    ap.add_argument("--sex_col", type=str, default="sex")
    ap.add_argument("--site_prefix", type=str, default="site_")

    ap.add_argument("--teacher_modality", type=str, default="fmri", choices=["smri", "fmri", "questionnaire"])
    ap.add_argument("--student_modality", type=str, default="smri", choices=["smri", "fmri", "questionnaire"])
    ap.add_argument("--deconfound", action="store_true")
    ap.add_argument("--teacher_cov_cols", type=str, default="sex,EstimatedTotalIntraCranialVol,MeanHeadMotionrfMRI")
    ap.add_argument("--teacher_cov_prefixes", type=str, default="site_")
    ap.add_argument("--student_cov_cols", type=str, default="sex,EstimatedTotalIntraCranialVol")
    ap.add_argument("--student_cov_prefixes", type=str, default="site_")

    ap.add_argument("--norm_holdout_frac", type=float, default=0.5)
    ap.add_argument("--k_folds", type=int, default=10)
    ap.add_argument("--inner_folds", type=int, default=5, help="Inner CV folds for the student's leakage-safe teacher OOF logits.")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--age_bins", type=int, default=10)
    ap.add_argument("--stratify_site", action="store_true")
    ap.add_argument("--stratify_sex", action="store_true")

    ap.add_argument("--age_bin_step", type=float, default=1.0)
    ap.add_argument("--age_grid_margin_bins", type=int, default=1)
    ap.add_argument("--label_sigma_s", type=float, default=1.0)
    ap.add_argument("--label_sigma_t", type=float, default=1.0)

    ap.add_argument("--activation", type=str, default="gelu")
    ap.add_argument("--no_layer_norm", action="store_true")
    ap.add_argument("--teacher_rep_hidden", type=str, default="32")
    ap.add_argument("--teacher_rep_dim", type=int, default=16)
    ap.add_argument("--teacher_head_hidden", type=str, default="")
    ap.add_argument("--student_rep_hidden", type=str, default="128,64")
    ap.add_argument("--student_rep_dim", type=int, default=64)
    ap.add_argument("--student_head_hidden", type=str, default="")

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)

    # KD settings for the distilled student stage (must match your pipeline.py run)
    ap.add_argument("--rkd_scale", type=float, default=50.0)
    ap.add_argument("--rkd_target", type=str, default="bag", choices=["bag", "expected_age"])
    ap.add_argument("--kd_method", type=str, default="rkd", choices=["rkd", "soft_label"])
    ap.add_argument("--distill_temperature", type=float, default=1.0)

    # the fixed lambda the STUDENT is tuned at
    ap.add_argument("--student_lambda", type=float, default=0.5,
                    help="Fixed lambda_kd for the student tuning stage (main result 0.5; sweep for sensitivity).")

    # frozen-teacher source for the student stage (option b)
    ap.add_argument("--teacher_config_json", type=str, default=None,
                    help="Path to a tune_best_configs.json holding the frozen teacher config "
                         "(default: <out_dir>/tune_best_configs.json).")
    ap.add_argument("--teacher_lr", type=float, default=None, help="Override: frozen teacher lr for the student stage.")
    ap.add_argument("--teacher_weight_decay", type=float, default=None, help="Override: frozen teacher weight_decay.")
    ap.add_argument("--teacher_dropout", type=float, default=None, help="Override: frozen teacher dropout.")

    # the three tuned hyperparameters (same grid for all stages)
    ap.add_argument("--lr_grid", type=str, default="3e-4,1e-3,3e-3")
    ap.add_argument("--weight_decay_grid", type=str, default="1e-4,1e-3")
    ap.add_argument("--dropout_grid", type=str, default="0.0,0.1,0.2,0.3")
    # 4th student axis, ACTIVE ONLY for kd_method='soft_label' AND student_lambda>0.
    # Under RKD (or lambda=0, or a single value) it is inert and the student stage
    # is identical to before: temperature is fixed at --distill_temperature.
    ap.add_argument("--distill_temperature_grid", type=str, default="1,2,5,10,20,50",
                    help="Soft-label-only: temperatures to sweep jointly with lr/wd/dropout "
                         "when kd_method=soft_label and student_lambda>0. Ignored under RKD.")

    ap.add_argument("--select_metric", type=str, default="mae", choices=["mae", "rmse", "r", "r2"],
                    help="HC-dev OOF metric to select on (default MAE; lower better for mae/rmse).")
    ap.add_argument("--stages", type=str, default="teacher,baseline,student",
                    help="Comma list from {teacher, baseline, student}. Runs in teacher->baseline->student order.")
    ap.add_argument("--seeds", type=str, default="123,456,789")
    ap.add_argument("--split_seed", type=int, default=140,
                    help="Fixed dev/holdout split seed. Default 140 == pipeline.py --seed 123 (123+17).")
    ap.add_argument("--device", type=str, default="cuda")

    args = ap.parse_args()

    if not (0.0 < args.norm_holdout_frac < 1.0):
        raise ValueError("--norm_holdout_frac must be in (0, 1).")
    lrs = [float(x) for x in parse_cols(args.lr_grid)]
    wds = [float(x) for x in parse_cols(args.weight_decay_grid)]
    drops = [float(x) for x in parse_cols(args.dropout_grid)]
    seeds = [int(x) for x in parse_cols(args.seeds)]
    stages = [s.strip().lower() for s in parse_cols(args.stages)]
    if not lrs or not wds or not drops or not seeds:
        raise ValueError("lr_grid / weight_decay_grid / dropout_grid / seeds must be non-empty.")
    if any(not (0.0 <= d < 1.0) for d in drops):
        raise ValueError("All dropout values must be in [0, 1).")
    if not (0.0 <= args.student_lambda <= 1.0):
        raise ValueError("--student_lambda must be in [0, 1].")
    if not set(stages) <= {"teacher", "baseline", "student"}:
        raise ValueError("--stages may only contain 'teacher', 'baseline', 'student'.")

    # Temperature is a real student axis ONLY under soft-label KD at lambda>0.
    # Under RKD it is mathematically inert (no distribution to sharpen), so we fix
    # it at --distill_temperature and the student stage behaves exactly as before.
    sweep_temp = (args.kd_method == "soft_label") and (args.student_lambda > 0.0)
    if sweep_temp:
        student_temps = [float(x) for x in parse_cols(args.distill_temperature_grid)]
        if not student_temps:
            raise ValueError("--distill_temperature_grid must be non-empty for soft_label at lambda>0.")
        if any(t <= 0.0 for t in student_temps):
            raise ValueError("All distill_temperature values must be > 0.")
    else:
        student_temps = [float(args.distill_temperature)]
    include_temp = sweep_temp and len(student_temps) > 1

    ensure_dir(args.out_dir)
    ctx, hc_dev_df = prepare(args)
    json_path = os.path.join(args.out_dir, "tune_best_configs.json")

    n_super = len(lrs) * len(wds) * len(drops) * len(seeds)
    n_student = n_super * len(student_temps)
    print("\n" + "=" * 90)
    print("TUNING PLAN")
    print(f"  teacher/baseline grid: {len(lrs)} lr x {len(wds)} wd x {len(drops)} dropout x {len(seeds)} seeds "
          f"= {n_super} runs each, {ctx.k_folds}-fold CV")
    if include_temp:
        print(f"  student grid: + {len(student_temps)} temperature (soft_label, lambda={args.student_lambda:g}) "
              f"= {n_student} runs, jointly selected over (lr, wd, dropout, T)")
    else:
        reason = ("RKD: temperature inert" if args.kd_method == "rkd"
                  else ("lambda=0: KD off" if args.student_lambda == 0.0 else "single fixed temperature"))
        print(f"  student grid: {n_super} runs ({reason}; temperature fixed at {args.distill_temperature:g})")
    print(f"  Stages: {stages} | student lambda = {args.student_lambda:g} | kd_method = {args.kd_method}")
    print(f"  Selection metric: pooled HC-dev OOF {args.select_metric.upper()} (mean over seeds)")
    print("=" * 90)

    best_configs: Dict[str, Any] = load_configs_json(json_path)
    best_configs.update({
        "selection_metric": args.select_metric,
        "seeds": seeds,
        "split_seed": args.split_seed,
        "corresponding_pipeline_seed": args.split_seed - 17,
        "student_lambda": args.student_lambda,
        "grids": {"lr": lrs, "weight_decay": wds, "dropout": drops},
        "kd": {"rkd_scale": args.rkd_scale, "rkd_target": args.rkd_target,
               "kd_method": args.kd_method, "distill_temperature": args.distill_temperature,
               "temperature_swept": bool(include_temp),
               "distill_temperature_grid": (student_temps if include_temp else None)},
        "fixed": {"label_sigma_s": args.label_sigma_s, "label_sigma_t": args.label_sigma_t,
                  "batch_size": args.batch_size, "inner_folds": args.inner_folds,
                  "teacher_rep_hidden": args.teacher_rep_hidden, "teacher_rep_dim": args.teacher_rep_dim,
                  "student_rep_hidden": args.student_rep_hidden, "student_rep_dim": args.student_rep_dim},
    })

    teacher_hp_in_memory: Optional[Dict[str, float]] = (
        {"lr": best_configs["teacher"]["lr"], "weight_decay": best_configs["teacher"]["weight_decay"],
         "dropout": best_configs["teacher"]["dropout"]}
        if "teacher" in best_configs and all(k in best_configs["teacher"] for k in ("lr", "weight_decay", "dropout"))
        else None
    )

    def save() -> None:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(best_configs, f, indent=2)

    # ============================ STAGE: TEACHER ============================
    if "teacher" in stages:
        print("\n[TEACHER] Tuning the privileged-modality supervised model — lr x weight_decay x dropout\n")
        df_t, sum_t = sweep(
            lambda lr, wd, dr, temp, seed: supervised_cv_oof(
                hc_dev_df, ctx, feat_cols=ctx.teacher_feat_cols, modality=ctx.teacher_modality,
                cfg=ctx.teacher_cfg(dr), sigma=ctx.label_sigma_t, is_teacher=True,
                lr=lr, weight_decay=wd, seed=seed),
            lrs=lrs, wds=wds, drops=drops, temps=[args.distill_temperature], seeds=seeds,
            select_metric=args.select_metric, label="teacher", include_temp=False)
        df_t.to_csv(os.path.join(args.out_dir, "tune_teacher_per_seed.csv"), index=False)
        sum_t.to_csv(os.path.join(args.out_dir, "tune_teacher_summary.csv"), index=False)
        print_summary(sum_t, args.select_metric, "[TEACHER] summary")
        best_configs["teacher"] = config_from_row(sum_t.iloc[0])
        teacher_hp_in_memory = {"lr": best_configs["teacher"]["lr"],
                                "weight_decay": best_configs["teacher"]["weight_decay"],
                                "dropout": best_configs["teacher"]["dropout"]}
        save()
        bt = best_configs["teacher"]
        print(f"\n[TEACHER] selected: lr={bt['lr']:g}, weight_decay={bt['weight_decay']:g}, dropout={bt['dropout']:g}")

    # ============================ STAGE: BASELINE ============================
    if "baseline" in stages:
        print("\n[BASELINE] Tuning the regular-modality supervised model (student @ lambda=0) — lr x weight_decay x dropout\n")
        df_b, sum_b = sweep(
            lambda lr, wd, dr, temp, seed: supervised_cv_oof(
                hc_dev_df, ctx, feat_cols=ctx.student_feat_cols, modality=ctx.student_modality,
                cfg=ctx.student_cfg(dr), sigma=ctx.label_sigma_s, is_teacher=False,
                lr=lr, weight_decay=wd, seed=seed),
            lrs=lrs, wds=wds, drops=drops, temps=[args.distill_temperature], seeds=seeds,
            select_metric=args.select_metric, label="baseline", include_temp=False)
        df_b.to_csv(os.path.join(args.out_dir, "tune_baseline_per_seed.csv"), index=False)
        sum_b.to_csv(os.path.join(args.out_dir, "tune_baseline_summary.csv"), index=False)
        print_summary(sum_b, args.select_metric, "[BASELINE] summary")
        best_configs["baseline"] = config_from_row(sum_b.iloc[0])
        save()
        bb = best_configs["baseline"]
        print(f"\n[BASELINE] selected: lr={bb['lr']:g}, weight_decay={bb['weight_decay']:g}, dropout={bb['dropout']:g}")

    # ============================ STAGE: STUDENT ============================
    if "student" in stages:
        teacher_hp = None
        if args.student_lambda > 0.0:
            teacher_hp = resolve_teacher_hp(args, teacher_hp_in_memory, args.out_dir)
            if teacher_hp is None:
                raise ValueError(
                    "student_lambda > 0 but no frozen teacher config could be resolved. "
                    "Run the 'teacher' stage first (persists to tune_best_configs.json), pass "
                    "--teacher_config_json, or give --teacher_lr/--teacher_weight_decay/--teacher_dropout."
                )
            print(f"\n[STUDENT] Using frozen teacher config from: {teacher_hp['source']} "
                  f"(lr={teacher_hp['lr']:g}, wd={teacher_hp['weight_decay']:g}, dropout={teacher_hp['dropout']:g})")
        else:
            print("\n[STUDENT] student_lambda == 0 -> pure supervised (coincides with baseline); no teacher needed.")

        axes = "lr x weight_decay x dropout" + (" x distill_temperature" if include_temp else "")
        print(f"\n[STUDENT] Tuning the distilled student at lambda={args.student_lambda:g} — {axes}\n")
        df_s, sum_s = sweep(
            lambda lr, wd, dr, temp, seed: student_distilled_cv_oof(
                hc_dev_df, ctx, lr=lr, weight_decay=wd, dropout=dr,
                student_lambda=args.student_lambda, distill_temperature=temp,
                teacher_hp=teacher_hp, seed=seed),
            lrs=lrs, wds=wds, drops=drops, temps=student_temps, seeds=seeds,
            select_metric=args.select_metric, label=f"student@{args.student_lambda:g}", include_temp=include_temp)
        tag = f"lambda{args.student_lambda:g}".replace(".", "p")
        df_s.to_csv(os.path.join(args.out_dir, f"tune_student_{tag}_per_seed.csv"), index=False)
        sum_s.to_csv(os.path.join(args.out_dir, f"tune_student_{tag}_summary.csv"), index=False)
        print_summary(sum_s, args.select_metric, f"[STUDENT lambda={args.student_lambda:g}] summary")
        student_cfg_sel = config_from_row(sum_s.iloc[0])
        student_cfg_sel["student_lambda"] = args.student_lambda
        if teacher_hp is not None:
            student_cfg_sel["teacher_config_used"] = {k: teacher_hp[k] for k in ("lr", "weight_decay", "dropout")}
        best_configs.setdefault("student", {})[f"lambda_{args.student_lambda:g}"] = student_cfg_sel
        save()
        bs = student_cfg_sel
        tsel = f", distill_temperature={bs['distill_temperature']:g}" if "distill_temperature" in bs else ""
        print(f"\n[STUDENT lambda={args.student_lambda:g}] selected: "
              f"lr={bs['lr']:g}, weight_decay={bs['weight_decay']:g}, dropout={bs['dropout']:g}{tsel}")

    # ============================ REPORT ============================
    print("\n" + "=" * 90)
    print(f"BEST CONFIGURATIONS  (selection metric = OOF {args.select_metric.upper()})")
    print("=" * 90)
    if "teacher" in best_configs:
        t = best_configs["teacher"]
        print("TEACHER:")
        print(f"  --teacher_lr {t['lr']:g}   --teacher_weight_decay {t['weight_decay']:g}   teacher dropout = {t['dropout']:g}")
        print(f"  OOF MAE {t['oof_mae_mean']:.3f}±{t['oof_mae_std']:.3f} | r {t['oof_r_mean']:.3f}")
    if "baseline" in best_configs:
        b = best_configs["baseline"]
        print("BASELINE (lambda=0):")
        print(f"  --baseline_lr {b['lr']:g}   --weight_decay {b['weight_decay']:g}   baseline dropout = {b['dropout']:g}")
        print(f"  OOF MAE {b['oof_mae_mean']:.3f}±{b['oof_mae_std']:.3f} | r {b['oof_r_mean']:.3f}")
    if "student" in best_configs and f"lambda_{args.student_lambda:g}" in best_configs["student"]:
        s = best_configs["student"][f"lambda_{args.student_lambda:g}"]
        print(f"STUDENT (lambda={args.student_lambda:g}, kd_method={args.kd_method}):")
        line = (f"  --student_lr {s['lr']:g}   --weight_decay {s['weight_decay']:g}   "
                f"student dropout = {s['dropout']:g}   --lambda_kd {args.student_lambda:g}   "
                f"--kd_method {args.kd_method}")
        if "distill_temperature" in s:
            line += f"   --distill_temperature {s['distill_temperature']:g}"
        print(line)
        print(f"  OOF MAE {s['oof_mae_mean']:.3f}±{s['oof_mae_std']:.3f} | r {s['oof_r_mean']:.3f}")

    # dropout-mismatch warning (pipeline.py single --dropout)
    dropouts = {}
    if "teacher" in best_configs: dropouts["teacher"] = best_configs["teacher"]["dropout"]
    if "baseline" in best_configs: dropouts["baseline"] = best_configs["baseline"]["dropout"]
    if "student" in best_configs and f"lambda_{args.student_lambda:g}" in best_configs["student"]:
        dropouts["student"] = best_configs["student"][f"lambda_{args.student_lambda:g}"]["dropout"]
    if len(set(dropouts.values())) > 1:
        print("\n  ** Selected dropouts differ across models:", dropouts)
        print("     pipeline.py uses ONE --dropout for teacher & student. Add separate")
        print("     --teacher_dropout / --student_dropout to pipeline.py to apply all selected values.")

    print(f"\nFrozen configs written to: {json_path}")
    print("Next: final pipeline.py runs (main result at lambda=0.5; then the lambda sensitivity sweep),")
    print(f"      using --seed {args.split_seed - 17} so the HC-dev/holdout split matches this tuning.")
    print("=" * 90)

if __name__ == "__main__":
    main()

'''
python tune.py \
  --smri_csv /data/00_Projects/distillation/data/04_ukb_distillation/multi_imaging/student.csv \
  --reg_csv /data/00_Projects/distillation/data/04_ukb_distillation/multi_imaging/teacher.csv \
  --target_csv /data/00_Projects/distillation/data/04_ukb_distillation/multi_imaging/UKB_health_and_cognition_status.csv \
  --out_dir /data/00_Projects/distillation/results/multi_imaging/paper/teacher_smri_student_fmri/soft_label_KD/tune_temperature/lam_0.5 \
  --hc_col is_healthy \
  --teacher_modality smri \
  --student_modality fmri \
  --deconfound \
  --teacher_rep_hidden 128,64 \
  --teacher_rep_dim 64 \
  --teacher_head_hidden "" \
  --student_rep_hidden 32,16 \
  --student_rep_dim 16 \
  --student_head_hidden "" \
  --label_sigma_t 1.0 \
  --label_sigma_s 1.0 \
  --lr_grid 3e-4 \
  --weight_decay_grid 1e-3 \
  --dropout_grid 0.4 \
  --seeds 123,456,789 \
  --split_seed 140 \
  --stages teacher,baseline,student \
  --student_lambda 0.5 \
  --kd_method soft_label \
  --select_metric mae \
  --k_folds 10 \
  --batch_size 64 \
  --max_epochs 200 \
  --patience 20 \
  --distill_temperature_grid 1,2,5,10,20,50
#  --device cuda
'''