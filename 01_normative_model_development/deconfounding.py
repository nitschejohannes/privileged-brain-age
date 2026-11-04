#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Sequence, Tuple
 
import numpy as np
import pandas as pd
from scipy.stats import rankdata, kruskal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
 
# ----------------------------------------------------------------------------
# Column roles
# ----------------------------------------------------------------------------
CONF_SEX = "sex"
CONF_ETIV = "EstimatedTotalIntraCranialVol"
CONF_MOTION = "MeanHeadMotionrfMRI"
SITE_DUMMIES = ["site_1", "site_2", "site_3"]          # all-zero => site 4 (reference)
 
QC_PREFIXES = ("InvertedSNR", "InvertedCNR", "Discrepancy", "AmountWarping")
 
def is_qc(col: str) -> bool:
    return any(col.startswith(p) for p in QC_PREFIXES)
 
def is_thickness(col: str) -> bool:
    return col.endswith("_thickness")
 
def split_smri_measures(cols: Sequence[str]) -> Tuple[List[str], List[str]]:
    """
    Partition sMRI columns into (volume-like, thickness).
    Volume-like = everything that is NOT *_thickness: the 32 subcortical volumes
    (Left-Hippocampus, Brain-Stem, CC_Anterior, ...) plus the 68 cortical
    *_volume columns. Thickness = the 68 *_thickness columns (no eTIV).
    """
    thickness = [c for c in cols if is_thickness(c)]
    volume_like = [c for c in cols if not is_thickness(c)]
    return volume_like, thickness

def _design(C: np.ndarray) -> np.ndarray:
    """Confound design matrix with a leading intercept column."""
    C = np.asarray(C, dtype=np.float64)
    if C.ndim == 1:
        C = C.reshape(-1, 1)
    return np.hstack([np.ones((C.shape[0], 1), dtype=np.float64), C])
 
def fit_residualizer(X_fit: np.ndarray, C_fit: np.ndarray) -> np.ndarray:
    """
    OLS betas for each feature regressed on the (intercept + confounds) design.
    Returns beta of shape (1 + n_confounds, n_features). Stores nothing else, so
    applying to new rows cannot depend on the fit sample beyond these betas.
    """
    D = _design(C_fit)
    beta, *_ = np.linalg.lstsq(D, np.asarray(X_fit, dtype=np.float64), rcond=None)
    return beta
 
def apply_residualizer(X: np.ndarray, C: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """
    Subtract the confound-predicted component using fit-set betas. The intercept
    is added back so features keep their fit-set location (mean) rather than
    being forced to zero on whatever subset is transformed.
    """
    D = _design(C)
    X = np.asarray(X, dtype=np.float64)
    return (X - D @ beta + beta[0:1, :]).astype(np.float64)

class BlockDeconfounder:
    """
    Residualizes a feature block. Different columns of the block can take
    different confound sets (used for sMRI: eTIV only on volume-like columns).
 
    groups : list of (column_names, confound_column_names)
        Each group's columns are residualized on that group's confounds. Groups
        must partition the block's columns with no overlap.
    """
 
    def __init__(self, feature_cols: Sequence[str], groups: List[Tuple[List[str], List[str]]]):
        self.feature_cols = list(feature_cols)
        self.col_index = {c: i for i, c in enumerate(self.feature_cols)}
        self.groups = groups
        self._betas: List[Tuple[np.ndarray, List[str], np.ndarray]] = []  # (mask, conf_cols, beta)
 
        covered = [c for cols, _ in groups for c in cols]
        if sorted(covered) != sorted(self.feature_cols):
            missing = set(self.feature_cols) - set(covered)
            extra = set(covered) - set(self.feature_cols)
            raise ValueError(f"groups must partition the block. missing={missing} extra={extra}")
 
    def fit(self, X_fit: np.ndarray, conf_fit: pd.DataFrame) -> "BlockDeconfounder":
        self._betas = []
        for cols, conf_cols in self.groups:
            mask = np.zeros(len(self.feature_cols), dtype=bool)
            for c in cols:
                mask[self.col_index[c]] = True
            C = conf_fit[conf_cols].to_numpy(np.float64) if conf_cols else None
            if C is None:
                self._betas.append((mask, [], None))
            else:
                self._betas.append((mask, list(conf_cols), fit_residualizer(X_fit[:, mask], C)))
        return self
 
    def transform(self, X: np.ndarray, conf: pd.DataFrame) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64).copy()
        for mask, conf_cols, beta in self._betas:
            if beta is None:
                continue
            C = conf[conf_cols].to_numpy(np.float64)
            X[:, mask] = apply_residualizer(X[:, mask], C, beta)
        return X
 
# ----------------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------------
def abs_spearman_vec(X: np.ndarray, c: np.ndarray) -> np.ndarray:
    """|Spearman rho| between every column of X and the vector c."""
    X = np.asarray(X, dtype=np.float64)
    cr = rankdata(c).astype(np.float64)
    cr -= cr.mean()
    cden = np.sqrt((cr ** 2).sum())
    out = np.zeros(X.shape[1])
    if cden == 0:
        return out
    for j in range(X.shape[1]):
        xj = rankdata(X[:, j]).astype(np.float64)
        xj -= xj.mean()
        denom = np.sqrt((xj ** 2).sum()) * cden
        out[j] = 0.0 if denom == 0 else abs(float((xj * cr).sum() / denom))
    return out
 
def site_label_from_dummies(conf: pd.DataFrame) -> np.ndarray:
    """Reconstruct an integer site label {1,2,3,4} from site_1..site_3 (all-zero => 4)."""
    lab = np.full(len(conf), 4, dtype=int)
    for j, c in enumerate(SITE_DUMMIES):
        lab[conf[c].to_numpy(np.float64) > 0.5] = j + 1
    return lab
 
def site_effect_vec(X: np.ndarray, site_label: np.ndarray) -> np.ndarray:
    """
    Per-feature rank-based effect size for a COLLECTIVE categorical site factor
    (4 levels), on the same ~[0, 1] scale as |Spearman rho|.
 
    epsilon-squared from the Kruskal-Wallis H statistic:
        eps2 = (H - k + 1) / (n - k),   k = number of groups
    clipped at 0, then sqrt'd so it is comparable in magnitude to |rho| (both
    are "fraction of rank variance" style measures). This reports site as ONE
    number per feature, matching the fact that site is corrected collectively
    (one design matrix with all three dummies), not dummy-by-dummy.
    """
    n = len(site_label)
    groups = [np.unique(site_label)]
    k = len(np.unique(site_label))
    out = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        samples = [X[site_label == g, j] for g in np.unique(site_label)]
        samples = [s for s in samples if len(s) > 0]
        if len(samples) < 2:
            continue
        H, _ = kruskal(*samples)
        eps2 = (H - k + 1) / max(n - k, 1)
        out[j] = float(np.sqrt(max(eps2, 0.0)))
    return out
 
def collect_group_values(
    X_before: np.ndarray, X_after: np.ndarray, conf: pd.DataFrame,
    continuous_confs: List[str],
) -> "Dict[str, Tuple[np.ndarray, np.ndarray]]":
    """
    Build the {confounder_label: (before_values, after_values)} dict used for the
    paired before/after boxplot: one entry per continuous/binary confounder
    (|Spearman rho| per feature) plus one collective 'site' entry (rank effect
    size per feature).
    """
    values: "Dict[str, Tuple[np.ndarray, np.ndarray]]" = {}
    for c in continuous_confs:
        cvec = conf[c].to_numpy(np.float64)
        values[c] = (abs_spearman_vec(X_before, cvec), abs_spearman_vec(X_after, cvec))
    site_label = site_label_from_dummies(conf)
    values["site"] = (site_effect_vec(X_before, site_label), site_effect_vec(X_after, site_label))
    return values
 
SHORT_LABEL = {
    CONF_ETIV: "TIV",
    CONF_SEX: "sex",
    CONF_MOTION: "motion",
    "site": "site",
}
 
def _render_panel(ax: "plt.Axes", values: "Dict[str, Tuple[np.ndarray, np.ndarray]]") -> float:
    """
    Draw one modality's paired before/after boxes onto an existing axis.
    Each confounder gets two x positions (before, after) with their own tick
    labels; the confounder name is placed as a second-level label underneath.
    Returns the max 'before' value plotted, so the caller can share y-limits.
    """
    confs = list(values.keys())
    before_c, after_c = "#B0B0B0", "#2ca02c"
    width = 0.62
    pair_gap = 0.18   # gap between the before/after box within a pair
    group_gap = 0.9   # extra gap between one confounder's pair and the next
    step = width + pair_gap + width + group_gap
    pos_b = [i * step for i in range(len(confs))]
    pos_a = [p + width + pair_gap for p in pos_b]
    centers = [(b + a) / 2 for b, a in zip(pos_b, pos_a)]
    data_b = [values[c][0] for c in confs]
    data_a = [values[c][1] for c in confs]
 
    def _box(data, pos, color):
        ax.boxplot(
            data, positions=pos, widths=width, patch_artist=True,
            whis=(5, 95), showfliers=False, manage_ticks=False,
            medianprops=dict(color="black", lw=1.3),
            boxprops=dict(facecolor=color, alpha=0.7, edgecolor="black", lw=0.8),
            whiskerprops=dict(color="black", lw=0.8), capprops=dict(color="black", lw=0.8),
        )
 
    rng = np.random.default_rng(0)
    for arr, p in zip(data_b, pos_b):
        ax.scatter(p + (rng.random(len(arr)) - 0.5) * width * 0.6, arr, s=3,
                  color=before_c, alpha=0.25, zorder=1, edgecolors="none")
    for arr, p in zip(data_a, pos_a):
        ax.scatter(p + (rng.random(len(arr)) - 0.5) * width * 0.6, arr, s=3,
                  color=after_c, alpha=0.30, zorder=1, edgecolors="none")
    _box(data_b, pos_b, before_c)
    _box(data_a, pos_a, after_c)
 
    # level 1: before/after ticks on the axis itself
    ax.set_xticks(pos_b + pos_a)
    ax.set_xticklabels(["before"] * len(confs) + ["after"] * len(confs), fontsize=7, rotation=0)
    ax.tick_params(axis="x", length=2)
 
    # level 2: confounder name centered under each before/after pair
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for c, x in zip(confs, centers):
        ax.text(x, -0.16, SHORT_LABEL.get(c, c), transform=trans,
                ha="center", va="top", fontsize=9, fontweight="bold")
 
    ax.set_xlim(pos_b[0] - 0.5, pos_a[-1] + 0.5)
    ax.grid(axis="y", alpha=0.25)
    return max(float(np.max(d)) for d in data_b)

def plot_all_groups(
    groups_values: "List[Tuple[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]]",
    out_path: str,
) -> None:
    n = len(groups_values)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.6), sharey=True)
    if n == 1:
        axes = [axes]
    maxes = []
    for ax, (label, values) in zip(axes, groups_values):
        maxes.append(_render_panel(ax, values))
        ax.set_title(label, fontsize=9, loc="left", pad=4)
    axes[0].set_ylabel("|ρ|  (site: √ε²)")
    ymax = min(1.0, max(0.35, max(maxes) * 1.1))
    axes[0].set_ylim(-0.02, ymax)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path, dpi=400)
    plt.close(fig)

def report_rows(group_name: str, X_before: np.ndarray, X_after: np.ndarray,
                conf: pd.DataFrame, confs: List[str]) -> List[dict]:
    rows = []
    for conf_name in confs:
        cvec = conf[conf_name].to_numpy(np.float64)
        for state, X in (("before", X_before), ("after", X_after)):
            rho = abs_spearman_vec(X, cvec)
            rows.append(dict(feature_group=group_name, state=state, confounder=conf_name,
                             mean_abs_rho=float(np.mean(rho)),
                             median_abs_rho=float(np.median(rho)),
                             p95_abs_rho=float(np.percentile(rho, 95)),
                             max_abs_rho=float(np.max(rho)), n_features=int(X.shape[1])))
    return rows

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smri_csv", required=True)
    ap.add_argument("--fmri_csv", required=True)
    ap.add_argument("--conf_csv", required=True)
    ap.add_argument("--eid_col", default="eid")
    ap.add_argument("--out_dir", default="deconf_out")
    ap.add_argument("--fit_ids", default=None,
                    help="Optional CSV with an eid column: rows to FIT residualizers on "
                         "(e.g. training eids). Coefficients are applied to ALL rows. "
                         "Omit to fit on all rows (single-file mode).")
    ap.add_argument("--no_plots", action="store_true",
                    help="Skip the before/after boxplots.")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
 
    smri = pd.read_csv(args.smri_csv)
    fmri = pd.read_csv(args.fmri_csv)
    conf = pd.read_csv(args.conf_csv)
 
    for name, d in (("smri", smri), ("fmri", fmri), ("conf", conf)):
        if args.eid_col not in d.columns:
            raise ValueError(f"{name} CSV is missing eid_col='{args.eid_col}'")
 
    conf_needed = [CONF_SEX, CONF_ETIV, CONF_MOTION] + SITE_DUMMIES
    missing_conf = [c for c in conf_needed if c not in conf.columns]
    if missing_conf:
        raise ValueError(f"confounder CSV missing columns: {missing_conf}")
 
    df = conf.merge(smri, on=args.eid_col, how="inner").merge(fmri, on=args.eid_col, how="inner")
 
    smri_cols = [c for c in smri.columns if c != args.eid_col]
    fmri_all = [c for c in fmri.columns if c != args.eid_col]
    fmri_feat = [c for c in fmri_all if not is_qc(c)]
    fmri_qc = [c for c in fmri_all if is_qc(c)]
 
    df = df.dropna(subset=[c for c in (conf_needed + smri_cols + fmri_feat) if c in df.columns]).reset_index(drop=True)
 
    volume_like, thickness = split_smri_measures(smri_cols)
    print(f"N (merged, complete) = {len(df)}")
    print(f"sMRI: {len(volume_like)} volume-like (eTIV+sex+site) | "
          f"{len(thickness)} thickness (sex+site, NO eTIV)")
    print(f"rs-fMRI: {len(fmri_feat)} connectivity edges (motion+sex+site) | "
          f"{len(fmri_qc)} QC columns quarantined")
 
    # fit mask (leakage control)
    if args.fit_ids:
        fit_eids = set(pd.read_csv(args.fit_ids)[args.eid_col].tolist())
        fit_mask = df[args.eid_col].isin(fit_eids).to_numpy()
        if fit_mask.sum() < 10:
            raise ValueError(f"--fit_ids matched only {int(fit_mask.sum())} rows.")
        print(f"Fitting residualizers on {int(fit_mask.sum())} / {len(df)} rows (from --fit_ids).")
    else:
        fit_mask = np.ones(len(df), dtype=bool)
        print("Fitting residualizers on ALL rows (single-file mode).")
 
    conf_df = df[conf_needed]
    conf_fit = conf_df.loc[fit_mask].reset_index(drop=True)
 
    # ---- sMRI ---------------------------------------------------------------
    Xs = df[smri_cols].to_numpy(np.float64)
    smri_groups = [
        (volume_like, [CONF_ETIV, CONF_SEX] + SITE_DUMMIES),   # eTIV + sex + site
        (thickness,   [CONF_SEX] + SITE_DUMMIES),              # sex + site only
    ]
    smri_dec = BlockDeconfounder(smri_cols, smri_groups).fit(Xs[fit_mask], conf_fit)
    Xs_corr = smri_dec.transform(Xs, conf_df)
 
    # ---- rs-fMRI ------------------------------------------------------------
    Xf = df[fmri_feat].to_numpy(np.float64)
    fmri_groups = [(fmri_feat, [CONF_MOTION, CONF_SEX] + SITE_DUMMIES)]
    fmri_dec = BlockDeconfounder(fmri_feat, fmri_groups).fit(Xf[fit_mask], conf_fit)
    Xf_corr = fmri_dec.transform(Xf, conf_df)
 
    # ---- write corrected tables --------------------------------------------
    out_smri = pd.concat([df[[args.eid_col]], pd.DataFrame(Xs_corr, columns=smri_cols)], axis=1)
    out_fmri = pd.concat([df[[args.eid_col]], pd.DataFrame(Xf_corr, columns=fmri_feat)], axis=1)
    out_qc = df[[args.eid_col] + fmri_qc].copy()
    out_smri.to_csv(os.path.join(args.out_dir, "smri_deconfounded.csv"), index=False)
    out_fmri.to_csv(os.path.join(args.out_dir, "fmri_deconfounded.csv"), index=False)
    out_qc.to_csv(os.path.join(args.out_dir, "fmri_qc.csv"), index=False)
 
    # ---- before/after report (evaluated on ALL rows) ------------------------
    rows: List[dict] = []
    Xs_vol_mask = np.array([c in set(volume_like) for c in smri_cols])
    rows += report_rows("sMRI_volume", Xs[:, Xs_vol_mask], Xs_corr[:, Xs_vol_mask],
                        conf_df, [CONF_ETIV, CONF_SEX] + SITE_DUMMIES)
    rows += report_rows("sMRI_thickness", Xs[:, ~Xs_vol_mask], Xs_corr[:, ~Xs_vol_mask],
                        conf_df, [CONF_ETIV, CONF_SEX] + SITE_DUMMIES)
    rows += report_rows("rsfMRI_edges", Xf, Xf_corr, conf_df,
                        [CONF_MOTION, CONF_SEX] + SITE_DUMMIES)
    rep = pd.DataFrame(rows)
    rep.to_csv(os.path.join(args.out_dir, "deconf_report.csv"), index=False)
 
    # ---- before/after boxplot (single figure, one panel per modality) -------
    if not args.no_plots:
        groups = [
            ("sMRI · volume", Xs[:, Xs_vol_mask], Xs_corr[:, Xs_vol_mask], [CONF_ETIV, CONF_SEX]),
            ("sMRI · thickness", Xs[:, ~Xs_vol_mask], Xs_corr[:, ~Xs_vol_mask], [CONF_ETIV, CONF_SEX]),
            ("rs-fMRI · across- and inside-network connectivity", Xf, Xf_corr, [CONF_MOTION, CONF_SEX]),
        ]
        groups_values = [
            (label, collect_group_values(Xb, Xa, conf_df, cont_confs))
            for label, Xb, Xa, cont_confs in groups
        ]
        fp = os.path.join(args.out_dir, "deconf_before_after.png")
        plot_all_groups(groups_values, fp)
        print(f"wrote {fp}")
 
    print("\n|Spearman rho| mean, before -> after (per feature group x confounder):")
    piv = rep.pivot_table(index=["feature_group", "confounder"], columns="state",
                          values="mean_abs_rho")[["before", "after"]]
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(piv.round(3))
    print(f"\nWrote corrected tables + deconf_report.csv to: {args.out_dir}")

if __name__ == "__main__":
    main()