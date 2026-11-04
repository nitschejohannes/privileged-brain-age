#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, norm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Column conventions
# -----------------------------------------------------------------------------
AGE = "age_at_assessment"
SEX = "sex"
SITE_COLS = ["site_1", "site_2", "site_3"]
TIV = "EstimatedTotalIntraCranialVol"
MOTION = "MeanHeadMotionrfMRI"

SYMPTOM_SUMS = ["phq9_sum", "gad7_sum", "audit_obs"]
ICD_COLS = ["MDD_icd10", "ANX_icd10", "AUD_icd10"]
COGNITIVE = ["fluid_intelligence", "reaction_time", "pairs_matching",
             "prospective_memory", "hand_grip_strength"]

# Model registry: (label, raw BAG col, bias-corrected BAG col, color)
@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    bag_raw: str
    bag_bc: str
    color: str

# insert teacher, student and baseline modality depending on the scenario
MODELS: List[ModelSpec] = [
    ModelSpec("teacher",  "Teacher (sMRI)",  "teacher_bag_raw",  "teacher_bag_biascorr",  "#2a9d8f"),
    ModelSpec("student",  "Student (rs-fMRI)",  "student_bag_raw",  "student_bag_biascorr",  "#e76f51"),
    ModelSpec("baseline", "Baseline (rs-fMRI)", "baseline_bag_raw", "baseline_bag_biascorr", "#264653"),
]

MIN_N = 10

# -----------------------------------------------------------------------------
# Stats helpers
# -----------------------------------------------------------------------------
def bh_fdr(pvals: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg q-values; NaNs pass through as NaN and are ignored."""
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return q
    idx = np.where(ok)[0]
    ps = p[idx]
    order = np.argsort(ps)
    ranked = ps[order]
    m = len(ranked)
    qraw = ranked * m / (np.arange(1, m + 1))
    qraw = np.minimum.accumulate(qraw[::-1])[::-1]
    qq = np.empty(m)
    qq[order] = np.clip(qraw, 0, 1)
    q[idx] = qq
    return q

def star(q: float) -> str:
    """Legacy thresholds used by Figures 3/4 (unchanged)."""
    if not np.isfinite(q):
        return ""
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    if q < 0.05:
        return "*"
    return ""

def star_ci(q: float) -> str:
    """Figure 1 thresholds: * q<0.05, ** q<0.01, *** q<=0.001."""
    if not np.isfinite(q):
        return ""
    if q <= 0.001:
        return "***"
    if q < 0.01:
        return "**"
    if q < 0.05:
        return "*"
    return ""

def pearson_ci(r: float, n: int, k: int = 0, alpha: float = 0.05) -> Tuple[float, float]:
    """Fisher z-transform CI for a Pearson correlation.

    k = number of confounders partialled out of one variable during
    residualization, so the effective df uses n - k - 3 (partial-correlation
    convention). Returns (lo, hi); (NaN, NaN) when unstable.
    """
    if r is None or not np.isfinite(r) or n is None:
        return (np.nan, np.nan)
    dof = n - k - 3
    if dof <= 0 or abs(r) >= 1.0:
        return (np.nan, np.nan)
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(dof)
    zc = norm.ppf(1.0 - alpha / 2.0)
    return (float(np.tanh(z - zc * se)), float(np.tanh(z + zc * se)))

def pearson_p_fisher(r: float, n: int, k: int = 0) -> float:
    """Two-sided p for r != 0 from the SAME Fisher-z SE used for the CI."""
    if r is None or not np.isfinite(r):
        return np.nan
    dof = n - k - 3
    if dof <= 0 or abs(r) >= 1.0:
        return np.nan
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(dof)
    return float(2.0 * norm.sf(abs(z) / se))

def bootstrap_corr_difference(
    df: pd.DataFrame,
    pair1: Tuple[str, str, str, str],
    pair2: Tuple[str, str, str, str],
    n_boot: int = 5000,
    random_state: int = 42,
) -> float:
    """One-sided paired bootstrap p-value for H1: r(pair1) > r(pair2).

    pair1 / pair2 are PAIRS entries (label, col_a, col_b, color). The SAME
    subjects are resampled for both correlations on one shared complete-case
    frame, so the test is paired and respects the covariance between the two
    correlations. Returns NaN when columns are missing or the frame is too small.
    """
    _, a1, b1, _ = pair1
    _, a2, b2, _ = pair2
    cols = list(dict.fromkeys([a1, b1, a2, b2]))
    if any(c not in df.columns for c in cols):
        return np.nan
    tmp = df[cols].dropna()
    n = len(tmp)
    if n < MIN_N:
        return np.nan
    rng = np.random.default_rng(random_state)
    x1 = tmp[a1].to_numpy(float); y1 = tmp[b1].to_numpy(float)
    x2 = tmp[a2].to_numpy(float); y2 = tmp[b2].to_numpy(float)
    boot_diffs = np.full(n_boot, np.nan)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xb1, yb1, xb2, yb2 = x1[idx], y1[idx], x2[idx], y2[idx]
        if min(np.std(xb1), np.std(yb1), np.std(xb2), np.std(yb2)) == 0:
            continue
        r1 = np.corrcoef(xb1, yb1)[0, 1]
        r2 = np.corrcoef(xb2, yb2)[0, 1]
        boot_diffs[i] = r1 - r2
    boot_diffs = boot_diffs[np.isfinite(boot_diffs)]
    if len(boot_diffs) == 0:
        return np.nan
    # One-sided p for H1: r1 > r2 (small p => pair1 more teacher-like than pair2).
    return float((np.sum(boot_diffs <= 0) + 1) / (len(boot_diffs) + 1))

def _design_matrix(frame: pd.DataFrame) -> np.ndarray:
    """Build the confounder design (with intercept) from whatever columns exist."""
    n = len(frame)
    cols: List[np.ndarray] = [np.ones(n)]
    if AGE in frame:
        age = frame[AGE].to_numpy(dtype=float)
        age_c = age - np.nanmean(age)
        cols.append(age_c)
        cols.append(age_c ** 2)
        if SEX in frame:
            sex = frame[SEX].to_numpy(dtype=float)
            cols.append(sex)
            cols.append(age_c * sex)
            cols.append(age_c ** 2 * sex)
    elif SEX in frame:
        cols.append(frame[SEX].to_numpy(dtype=float))
    for s in SITE_COLS:
        if s in frame:
            cols.append(frame[s].to_numpy(dtype=float))
    if TIV in frame:
        cols.append(frame[TIV].to_numpy(dtype=float))
    if MOTION in frame:
        cols.append(frame[MOTION].to_numpy(dtype=float))
    X = np.column_stack(cols)
    # Drop zero-variance columns (keep intercept at position 0).
    keep = [0]
    for j in range(1, X.shape[1]):
        if np.nanstd(X[:, j]) > 1e-12:
            keep.append(j)
    return X[:, keep]

def n_confound_cols(frame: pd.DataFrame) -> int:
    """Number of confounders actually partialled (design columns minus intercept)."""
    return int(_design_matrix(frame).shape[1] - 1)

def residualize(y: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
    """Return OLS residuals of y on the confounder design built from `frame`."""
    y = np.asarray(y, dtype=float)
    X = _design_matrix(frame)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta

def _corr(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan, np.nan, np.nan, np.nan
    r, p = pearsonr(x, y)
    rho, prho = spearmanr(x, y)
    return float(r), float(p), float(rho), float(prho)

def _common_frame(df: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """Common complete-case frame: outcome + all model BAGs + present confounders."""
    bag_cols = [m.bag_bc for m in MODELS]
    conf_present = [c for c in ([AGE, SEX, TIV, MOTION] + SITE_COLS) if c in df.columns]
    need = [c for c in ([outcome] + bag_cols + conf_present) if c in df.columns]
    return df.dropna(subset=need).copy()

def corr_models_vs_outcome(
    df: pd.DataFrame, outcome: str, *, residualized: bool = True
) -> List[Dict]:
    """
    Correlate each model's BAG with `outcome` on ONE common complete-case frame.
    If residualized, use the bias-corrected BAG residualized on confounders; else
    use the bias-corrected BAG as-is. Also returns n_confounds (k), identical
    across models because they share the frame and design.
    """
    frame = _common_frame(df, outcome)
    n = len(frame)
    k = n_confound_cols(frame) if (n >= MIN_N and residualized) else 0
    out: List[Dict] = []
    for m in MODELS:
        if n < MIN_N or m.bag_bc not in frame:
            out.append(dict(model=m.key, label=m.label, r=np.nan, p=np.nan,
                            rho=np.nan, prho=np.nan, n=n, n_confounds=k))
            continue
        bag = frame[m.bag_bc].to_numpy(float)
        bag = residualize(bag, frame) if residualized else bag
        r, p, rho, prho = _corr(bag, frame[outcome].to_numpy(float))
        out.append(dict(model=m.key, label=m.label, r=r, p=p, rho=rho, prho=prho,
                        n=n, n_confounds=k))
    return out

def corr_pair_raw(df: pd.DataFrame, col_a: str, col_b: str) -> Dict:
    """Raw-BAG correlation between two models on their common complete-case frame."""
    need = [c for c in (col_a, col_b) if c in df.columns]
    if len(need) < 2:
        return dict(r=np.nan, p=np.nan, rho=np.nan, prho=np.nan, n=0)
    frame = df.dropna(subset=need)
    n = len(frame)
    if n < MIN_N:
        return dict(r=np.nan, p=np.nan, rho=np.nan, prho=np.nan, n=n)
    r, p, rho, prho = _corr(frame[col_a].to_numpy(float), frame[col_b].to_numpy(float))
    return dict(r=r, p=p, rho=rho, prho=prho, n=n)

def bootstrap_ci_r(
    frame: pd.DataFrame, bag_col: str, outcome: str, n_boot: int,
    rng: np.random.Generator, alpha: float = 0.05,
) -> Tuple[float, float]:
    """Percentile bootstrap CI for corr(residualized BAG, outcome).

    The residualization is part of the estimator, so it is re-fit on every
    resample. Returns (lo, hi); (NaN, NaN) if too few valid resamples.
    """
    m = len(frame)
    if m < MIN_N:
        return (np.nan, np.nan)
    rs = np.full(n_boot, np.nan)
    for b in range(n_boot):
        take = rng.integers(0, m, m)
        fb = frame.iloc[take]
        try:
            bres = residualize(fb[bag_col].to_numpy(float), fb)
            yv = fb[outcome].to_numpy(float)
            if np.nanstd(bres) == 0 or np.nanstd(yv) == 0:
                continue
            rs[b] = np.corrcoef(bres, yv)[0, 1]
        except Exception:
            continue
    rs = rs[np.isfinite(rs)]
    if len(rs) < 20:
        return (np.nan, np.nan)
    lo = np.percentile(rs, 100 * alpha / 2.0)
    hi = np.percentile(rs, 100 * (1.0 - alpha / 2.0))
    return (float(lo), float(hi))

# -----------------------------------------------------------------------------
# Derived severity variables
# -----------------------------------------------------------------------------
def add_severity_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "phq9_sum" in df:
        df["PHQ9_ge_10"] = (df["phq9_sum"] >= 10).astype("float")
        df["PHQ9_10_to_15"] = ((df["phq9_sum"] >= 10) & (df["phq9_sum"] < 15)).astype("float")
        df["PHQ9_15_to_20"] = ((df["phq9_sum"] >= 15) & (df["phq9_sum"] < 20)).astype("float")
        df["PHQ9_gt_20"] = (df["phq9_sum"] >= 20).astype("float")
    if "gad7_sum" in df:
        df["GAD7_ge_10"] = (df["gad7_sum"] >= 10).astype("float")
        df["GAD7_10_to_15"] = ((df["gad7_sum"] >= 10) & (df["gad7_sum"] < 15)).astype("float")
        df["GAD7_15_to_20"] = ((df["gad7_sum"] >= 15) & (df["gad7_sum"] < 20)).astype("float")
        df["GAD7_gt_20"] = (df["gad7_sum"] >= 20).astype("float")
    if "audit_obs" in df:
        df["AUDIT_ge_4"] = (df["audit_obs"] >= 4).astype("float")
        df["AUDIT_ge_8"] = (df["audit_obs"] >= 8).astype("float")
    return df

# -----------------------------------------------------------------------------
# Figure: symptom associations in matched ICD subgroups (now with 95% CIs)
# -----------------------------------------------------------------------------
MATCHED = [  # (sum score, ICD flag, panel title)
    ("phq9_sum", "MDD_icd10", "PHQ-9 in ICD-10 MDD"),
    ("gad7_sum", "ANX_icd10", "GAD-7 in ICD-10 ANX"),
    ("audit_obs", "AUD_icd10", "AUDIT-C in ICD-10 AUD"),
]

def figure_matched_symptoms(
    cases: pd.DataFrame, out_dir: str, *,
    ci_method: str = "fisher", n_boot: int = 2000, seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: List[Dict] = []
    for sum_col, icd_col, title in MATCHED:
        if sum_col not in cases or icd_col not in cases:
            continue
        sub = cases[cases[icd_col] == 1]
        frame = _common_frame(sub, sum_col)  # for optional bootstrap
        res = corr_models_vs_outcome(sub, sum_col, residualized=True)
        for r in res:
            n = int(r.get("n", 0))
            k = int(r.get("n_confounds", 0))
            rr = r["r"]
            lo, hi = pearson_ci(rr, n, k)
            pf = pearson_p_fisher(rr, n, k)
            if ci_method == "bootstrap" and n_boot > 0 and np.isfinite(rr):
                m = next(mm for mm in MODELS if mm.key == r["model"])
                blo, bhi = bootstrap_ci_r(frame, m.bag_bc, sum_col, n_boot, rng)
                if np.isfinite(blo) and np.isfinite(bhi):
                    lo, hi = blo, bhi
            r.update(panel=title, outcome=sum_col, subgroup=icd_col,
                     ci_lo=lo, ci_hi=hi, ci_method=ci_method, p_fisher=pf)
            rows.append(r)
    stats = pd.DataFrame(rows)
    if not stats.empty:
        # Stars come from BH-FDR over the Fisher-z p (coherent with the CI).
        stats["q"] = bh_fdr(stats["p_fisher"].to_numpy())
        # Retain the original pearsonr-based q for reference.
        stats["q_pearson"] = bh_fdr(stats["p"].to_numpy())
    _grouped_bar_with_ci(
        stats,
        #title="Corrected BAG vs symptom severity (matched ICD-10 subgroups)",
        ylabel="Pearson r (BAG residualized), 95% CI",
        out_path=os.path.join(out_dir, "matched_symptoms.png"),
    )
    stats.to_csv(os.path.join(out_dir, "matched_symptoms_stats.csv"), index=False)
    return stats

# -----------------------------------------------------------------------------
# Figure: cognition, HC (primary) and cases (secondary), not pooled
# -----------------------------------------------------------------------------
def figure_cognition(hc: pd.DataFrame, cases: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    groups = [("HC (primary)", hc), ("Cases (secondary)", cases)]
    cog_present = [c for c in COGNITIVE if c in hc.columns or c in cases.columns]
    fig, axes = plt.subplots(len(groups), 1, figsize=(1.6 * max(len(cog_present), 1) + 3, 8), squeeze=False)
    rows: List[Dict] = []
    for ax, (gname, gdf) in zip(axes[:, 0], groups):
        panel_rows: List[Dict] = []
        for cog in cog_present:
            res = corr_models_vs_outcome(gdf, cog, residualized=True)
            for r in res:
                r.update(group=gname, outcome=cog)
                panel_rows.append(r); rows.append(r)
        pstats = pd.DataFrame(panel_rows)
        if not pstats.empty:
            pstats["q"] = bh_fdr(pstats["p"].to_numpy())
        _draw_grouped_bars_on_ax(ax, pstats, panel_col="outcome", value="r")
        ax.set_title(f"Corrected BAG vs cognition — {gname}")
        ax.set_ylabel("Pearson r (BAG residualized)")

        # --- headroom so significance markers sit INSIDE the frame ---
        lo, hi = ax.get_ylim()
        span = hi - lo if hi > lo else 1.0
        ax.set_ylim(lo, hi + 0.15 * span)   # ~15% top padding for the asterisks

    # --- single shared legend: capture handles from any populated panel,
    #     then remove every per-axis legend (version-agnostic) ---
    handles, labels = [], []
    for ax in axes[:, 0]:
        h, l = ax.get_legend_handles_labels()
        if h and not handles:
            handles, labels = h, l
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    if handles:
        fig.legend(handles, labels, loc="upper center",
                   ncol=len(labels), frameon=False,
                   bbox_to_anchor=(0.5, 1.0))

    fig.tight_layout(rect=(0, 0, 1, 0.96))           # leave room for the top legend
    fig.savefig(os.path.join(out_dir, "fig3_cognition.png"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    stats = pd.DataFrame(rows)
    if not stats.empty:
        stats["q"] = stats.groupby("group")["p"].transform(lambda s: bh_fdr(s.to_numpy()))
    stats.to_csv(os.path.join(out_dir, "fig3_cognition_stats.csv"), index=False)
    return stats

# -----------------------------------------------------------------------------
# Figure: cross-model RAW BAG correlations across HC / MDD / GAD / AUD
# -----------------------------------------------------------------------------
PAIRS = [  # (label, col_a, col_b, color)
    ("Student-Teacher",  "student_bag_raw",  "teacher_bag_raw",  "#00798c"),
    ("Baseline-Teacher", "baseline_bag_raw", "teacher_bag_raw",  "#616161"),
    ("Student-Baseline", "student_bag_raw",  "baseline_bag_raw", "#6a4c93"),
]

# Point layout within each group (same order as PAIRS above).
_PAIR_OFFSETS = np.array([-0.22, 0.0, 0.22])

FIG4_YFLOOR = 0.5
FIG4_YMAX_CAP = 1.12

CROSS_MODEL_TESTS = [
    ("ST > BT", 0, 1),  # r(Student,Teacher)  > r(Baseline,Teacher)
    ("ST > SB", 0, 2),  # r(Student,Teacher)  > r(Student,Baseline)
]

def _sig_bracket(ax, x1: float, x2: float, y: float, label: str, *,
                 height: float = 0.02, lw: float = 1.1, fontsize: float = 10,
                 color: str = "black") -> None:
    """Draw a single upward-opening significance bracket with its label centred
    above the connecting bar."""
    xl, xr = (x1, x2) if x1 <= x2 else (x2, x1)
    ax.plot([xl, xl, xr, xr], [y, y + height, y + height, y],
            color=color, lw=lw, clip_on=False, zorder=5)
    ax.text((xl + xr) / 2.0, y + height, label,
            ha="center", va="bottom", fontsize=fontsize, zorder=6)

def _place_group_brackets(ax, brackets: List[Tuple[float, float, str]], y0: float,
                          *, step: float, inset: float = 0.02, height: float = 0.02) -> float:
    """Lay out a group's significance brackets without overlap.

    `brackets` is a list of (x_left, x_right, star_label). Brackets that would
    overlap horizontally are pushed onto higher tiers (vertical stacking); each
    tier additionally insets its arms inward so that stacked brackets sharing an
    endpoint do not draw their vertical arms on top of one another (horizontal
    offset). Narrower comparisons are placed on the lower tiers so nested
    comparisons read inside-out. Returns the y of the highest bracket drawn.
    """
    if not brackets:
        return y0
    # Narrow (inner) comparisons first so they sit on the lowest tiers.
    ordered = sorted(brackets, key=lambda b: (abs(b[1] - b[0]), min(b[0], b[1])))
    placed: List[Tuple[int, float, float]] = []  # (tier, xL, xR) of drawn brackets
    top_y = y0
    for x1, x2, label in ordered:
        xl, xr = (x1, x2) if x1 <= x2 else (x2, x1)
        # Lowest tier with no horizontal overlap against brackets already there.
        tier = 0
        while any(t == tier and not (xr < pL or xl > pR) for t, pL, pR in placed):
            tier += 1
        d = inset * tier                      # per-tier horizontal inset
        xa, xb = xl + d, xr - d
        if xb <= xa:                          # never invert a very narrow bracket
            xa, xb = xl, xr
        y = y0 + tier * step
        _sig_bracket(ax, xa, xb, y, label, height=height)
        placed.append((tier, xl, xr))
        top_y = max(top_y, y + height)
    return top_y

def figure_cross_model(
    hc: pd.DataFrame, cases: pd.DataFrame, out_dir: str, *,
    n_boot: int = 5000, seed: int = 0,
    ymin: Optional[float] = None, ymax: Optional[float] = None,
) -> pd.DataFrame:
    """y-axis is BROKEN: a 0 stub at the bottom (with a diagonal break mark) then
    the data band from FIG4_YFLOOR (0.5) up. Lower limit / break point defaults to
    FIG4_YFLOOR so all groups share one reproducible axis; the top auto-fits to the
    tallest bracket but is capped at FIG4_YMAX_CAP. Pass ymin / ymax to override
    either end (ymin=None -> fixed floor, ymax=None -> auto top with cap)."""
    groups: List[Tuple[str, pd.DataFrame]] = [("HC", hc)]
    for icd, name in [("MDD_icd10", "MDD"), ("ANX_icd10", "GAD/ANX"), ("AUD_icd10", "AUD")]:
        if icd in cases.columns:
            groups.append((name, cases[cases[icd] == 1]))
    group_names = [g for g, _ in groups]

    # --- correlations + Fisher-z CIs (raw BAGs, so k = 0 confounders) ---------
    rows: List[Dict] = []
    for gname, gdf in groups:
        for label, a, b, _ in PAIRS:
            res = corr_pair_raw(gdf, a, b)
            lo, hi = pearson_ci(res["r"], res["n"], k=0)
            res.update(group=gname, pair=label, ci_lo=lo, ci_hi=hi)
            rows.append(res)
    stats = pd.DataFrame(rows)
    if not stats.empty:
        stats["q"] = bh_fdr(stats["p"].to_numpy())

    # --- directional paired-bootstrap tests per group -------------------------
    test_rows: List[Dict] = []
    for gi, (gname, gdf) in enumerate(groups):
        for ti, (tlabel, hi_idx, lo_idx) in enumerate(CROSS_MODEL_TESTS):
            p = bootstrap_corr_difference(
                gdf, PAIRS[hi_idx], PAIRS[lo_idx],
                n_boot=n_boot, random_state=seed + 1000 * gi + ti,
            )
            test_rows.append(dict(group=gname, test=tlabel,
                                  hi_idx=hi_idx, lo_idx=lo_idx, p=p))
    test_df = pd.DataFrame(test_rows)
    if not test_df.empty:
        test_df["q"] = bh_fdr(test_df["p"].to_numpy())  # FDR over the test family

    # --- plot: broken y-axis (a 0 stub at the bottom, then the data band) ------
    xbase = np.arange(len(group_names))
    pair_x = {label: xbase + _PAIR_OFFSETS[i] for i, (label, *_ ) in enumerate(PAIRS)}
    break_low = FIG4_YFLOOR if ymin is None else ymin

    fig, (ax, ax0) = plt.subplots(
        2, 1, sharex=True,
        figsize=(1.7 * len(group_names) + 3, 5.2),
        gridspec_kw={"height_ratios": [20, 1], "hspace": 0.06},
    )

    # points + asymmetric CI whiskers on the main (upper) axis
    for i, (label, a, b, color) in enumerate(PAIRS):
        xs = pair_x[label]
        vals, elo, ehi = [], [], []
        for gname in group_names:
            row = stats[(stats["group"] == gname) & (stats["pair"] == label)]
            if len(row):
                r = float(row["r"].iloc[0])
                lo = float(row["ci_lo"].iloc[0]); hi = float(row["ci_hi"].iloc[0])
            else:
                r = lo = hi = np.nan
            vals.append(r)
            elo.append(r - lo if np.isfinite(lo) else np.nan)
            ehi.append(hi - r if np.isfinite(hi) else np.nan)
        ax.errorbar(xs, vals, yerr=[elo, ehi], fmt="o", color=color, markersize=6,
                    capsize=4, capthick=1.2, lw=1.5, label=label, zorder=3)

    # significance brackets: compressed "bar above" style, only FDR-significant,
    # stacked + horizontally inset so many significant results never collide.
    step = 0.045     # compressed vertical spacing between stacked brackets
    base_pad = 0.03  # gap above the tallest CI before the first bracket
    b_height = 0.014
    bracket_top = -np.inf  # highest bracket drawn, so the top limit leaves room
    for j, gname in enumerate(group_names):
        gs = stats[stats["group"] == gname]
        # baseline = just above the tallest CI whisker (or point) in the group
        base = np.nanmax(np.concatenate([
            gs["ci_hi"].to_numpy(dtype=float),
            gs["r"].to_numpy(dtype=float),
        ]))
        if not np.isfinite(base):
            continue
        y0 = base + base_pad
        brackets: List[Tuple[float, float, str]] = []
        for _, tr in test_df[test_df["group"] == gname].iterrows():
            s = star(tr["q"])
            if not s:
                continue
            x_hi = pair_x[PAIRS[int(tr["hi_idx"])][0]][j]
            x_lo = pair_x[PAIRS[int(tr["lo_idx"])][0]][j]
            brackets.append((x_hi, x_lo, s))
        top = _place_group_brackets(ax, brackets, y0, step=step, height=b_height)
        bracket_top = max(bracket_top, top)

    # y-limits: main axis spans [break_low, top]; the star labels sit just above
    # the highest bracket, and the top is CAPPED so brackets can't balloon it.
    data_lo = np.nanmin(np.concatenate([
        stats["ci_lo"].to_numpy(dtype=float), stats["r"].to_numpy(dtype=float),
    ])) if not stats.empty else np.nan
    data_hi = np.nanmax(np.concatenate([
        stats["ci_hi"].to_numpy(dtype=float), stats["r"].to_numpy(dtype=float),
        np.array([bracket_top if np.isfinite(bracket_top) else np.nan]),
    ])) if not stats.empty else np.nan
    top = min(data_hi + 0.03, FIG4_YMAX_CAP) if np.isfinite(data_hi) else 1.05
    if ymax is not None:
        top = ymax
    ax.set_ylim(break_low, top)
    ax0.set_ylim(0.0, max(break_low * 0.06, 1e-3))  # tiny stub that contains 0

    # Guard: the break point can clip a CI whisker that dips below it -- warn
    # loudly rather than silently cropping a confidence interval.
    if np.isfinite(data_lo) and data_lo < break_low - 1e-9:
        print(f"[fig4] WARNING: broken-axis floor {break_low:g} clips a CI/point that "
              f"reaches {data_lo:.3f}. Lower --fig4_ymin (or FIG4_YFLOOR) to show it.")

    # --- broken-axis cosmetics ------------------------------------------------
    ax.spines["bottom"].set_visible(False)
    ax0.spines["top"].set_visible(False)
    # keep only the left y-axis and the bottom x-axis (no top/right frame)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax0.spines["right"].set_visible(False)
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax0.set_yticks([0.0])
    ax0.axhline(0.0, color="#BDBDBD", lw=0.8, zorder=1)  # 0 reference in the stub

    # diagonal break marks straddling the gap between the two axes
    dbreak = 0.7
    kw = dict(marker=[(-1, -dbreak), (1, dbreak)], markersize=7, linestyle="none",
              color="k", mec="k", mew=1, clip_on=False)
    ax.plot([0], [0], transform=ax.transAxes, **kw)   # break mark on the left only
    ax0.plot([0], [1], transform=ax0.transAxes, **kw)  # break mark on the left only

    ax0.set_xticks(xbase); ax0.set_xticklabels(group_names)
    fig.supylabel("Pearson r (raw BAGs)", x=0.03, fontsize=11)
    #ax.set_title("Cross-model raw-BAG correlation (teacher-likeness) by group")
    ax.legend(
        fontsize=10,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(PAIRS),
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig4_cross_model.png"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    # --- merge directional test results into the stats table for the CSV ------
    if not test_df.empty:
        wide = test_df.pivot(index="group", columns="test", values=["p", "q"])
        wide.columns = [f"{stat}_{tst.replace(' ', '_').replace('>', 'gt')}"
                        for stat, tst in wide.columns]
        stats = stats.merge(wide.reset_index(), on="group", how="left")
    stats.to_csv(os.path.join(out_dir, "fig4_cross_model_stats.csv"), index=False)
    return stats

def _draw_grouped_bars_on_ax(ax, stats: pd.DataFrame, *, panel_col: str, value: str) -> None:
    if stats.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return
    panels = list(dict.fromkeys(stats[panel_col].tolist()))
    xbase = np.arange(len(panels))
    width = 0.8 / len(MODELS)
    for i, m in enumerate(MODELS):
        vals, stars_, ns = [], [], []
        for pnl in panels:
            row = stats[(stats[panel_col] == pnl) & (stats["model"] == m.key)]
            v = float(row[value].iloc[0]) if len(row) else np.nan
            q = float(row["q"].iloc[0]) if len(row) and "q" in row.columns else np.nan
            n = int(row["n"].iloc[0]) if len(row) and "n" in row.columns else 0
            vals.append(v); stars_.append(star(q)); ns.append(n)
        xs = xbase + (i - (len(MODELS) - 1) / 2) * width
        ax.bar(xs, vals, width=width, color=m.color, label=m.label)
        for x, v, s in zip(xs, vals, stars_):
            if np.isfinite(v):
                ax.text(x, v + (0.02 if v >= 0 else -0.05), s, ha="center", fontsize=11)
    ax.axhline(0.0, color="#BDBDBD", lw=0.8)
    ax.set_xticks(xbase)
    ax.set_xticklabels([f"{p}\n(n={int(stats[stats[panel_col]==p]['n'].max())})" for p in panels], fontsize=11)
    ax.legend(fontsize=10)

def _grouped_bar_with_ci(stats: pd.DataFrame, *,# title: str,
                         ylabel: str, out_path: str,
                         panel_col: str = "panel") -> None:
    """Grouped bars (models within panels) with 95% CI whiskers and BH asterisks
    placed over the CI. Expects columns: r, ci_lo, ci_hi, q, n."""
    if stats.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        fig.savefig(out_path, dpi=600, bbox_inches="tight")
        plt.close(fig)
        return

    panels = list(dict.fromkeys(stats[panel_col].tolist()))
    xbase = np.arange(len(panels))
    width = 0.8 / len(MODELS)
    fig, ax = plt.subplots(figsize=(2.6 * max(len(panels), 1) + 2, 4.8))

    all_hi: List[float] = []
    all_lo: List[float] = []
    for i, m in enumerate(MODELS):
        vals, los, his, stars_ = [], [], [], []
        for pnl in panels:
            row = stats[(stats[panel_col] == pnl) & (stats["model"] == m.key)]
            v = float(row["r"].iloc[0]) if len(row) else np.nan
            lo = float(row["ci_lo"].iloc[0]) if len(row) and "ci_lo" in row.columns else np.nan
            hi = float(row["ci_hi"].iloc[0]) if len(row) and "ci_hi" in row.columns else np.nan
            q = float(row["q"].iloc[0]) if len(row) and "q" in row.columns else np.nan
            vals.append(v); los.append(lo); his.append(hi); stars_.append(star_ci(q))
        xs = xbase + (i - (len(MODELS) - 1) / 2) * width
        ax.bar(xs, vals, width=width, color=m.color, label=m.label, zorder=2)

        # asymmetric CI whiskers (Fisher-z / bootstrap intervals are not symmetric in r)
        xs_e, ys_e, e_lo, e_hi = [], [], [], []
        for x, v, lo, hi in zip(xs, vals, los, his):
            if np.isfinite(v) and np.isfinite(lo) and np.isfinite(hi):
                xs_e.append(x); ys_e.append(v)
                e_lo.append(max(v - lo, 0.0)); e_hi.append(max(hi - v, 0.0))
                all_hi.append(hi); all_lo.append(lo)
        if xs_e:
            ax.errorbar(xs_e, ys_e, yerr=[e_lo, e_hi], fmt="none",
                        ecolor="#2b2b2b", elinewidth=1.1, capsize=3, zorder=3)

        # asterisks placed just beyond the CI whisker, in the bar's direction
        for x, v, lo, hi, s in zip(xs, vals, los, his, stars_):
            if not s or not np.isfinite(v):
                continue
            if v >= 0 and np.isfinite(hi):
                ax.text(x, hi, s, ha="center", va="bottom", fontsize=10, zorder=4)
            elif v < 0 and np.isfinite(lo):
                ax.text(x, lo, s, ha="center", va="top", fontsize=10, zorder=4)

    ax.axhline(0.0, color="#BDBDBD", lw=0.8)
    ax.set_xticks(xbase)
    labels = []
    for p in panels:
        sub = stats[stats[panel_col] == p]
        nmax = int(sub["n"].max()) if "n" in sub.columns and sub["n"].notna().any() else 0
        labels.append(f"{p}\n(n={nmax})")
    ax.set_xticklabels(labels, fontsize=12)
    #ax.set_title(title)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(
        fontsize=10,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(MODELS),
        frameon=False,
    )

    if all_hi or all_lo:
        top = max(all_hi + [0.0]); bot = min(all_lo + [0.0])
        span = top - bot if top > bot else 1.0
        pad = 0.10 * span
        ax.set_ylim(bot - pad, top + pad)

    fig.tight_layout()
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

# -----------------------------------------------------------------------------
# Data-availability table
# -----------------------------------------------------------------------------
def availability_table(hc: pd.DataFrame, cases: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    allsub = pd.concat([hc, cases], ignore_index=True)
    groups = {"all": allsub, "cases": cases, "hc": hc}
    conf_present = [c for c in ([AGE, SEX, TIV, MOTION] + SITE_COLS)]
    bag_cols = [m.bag_bc for m in MODELS]
    outcomes = SYMPTOM_SUMS + COGNITIVE
    rows: List[Dict] = []
    for gname, gdf in groups.items():
        for out in outcomes:
            if out not in gdf.columns:
                rows.append(dict(group=gname, outcome=out, n_total=len(gdf),
                                 n_outcome=0, n_complete_case=0))
                continue
            n_out = int(gdf[out].notna().sum())
            need = [out] + [c for c in bag_cols if c in gdf.columns] + [c for c in conf_present if c in gdf.columns]
            n_cc = int(gdf.dropna(subset=need).shape[0])
            rows.append(dict(group=gname, outcome=out, n_total=len(gdf),
                             n_outcome=n_out, n_complete_case=n_cc))
    # matched-subgroup availability (symptom in its ICD subgroup)
    for sum_col, icd_col, title in MATCHED:
        if sum_col in cases.columns and icd_col in cases.columns:
            sub = cases[cases[icd_col] == 1]
            need = [sum_col] + [c for c in bag_cols if c in sub.columns] + [c for c in conf_present if c in sub.columns]
            rows.append(dict(group=f"cases:{icd_col}==1", outcome=sum_col,
                             n_total=len(sub), n_outcome=int(sub[sum_col].notna().sum()),
                             n_complete_case=int(sub.dropna(subset=need).shape[0])))
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(out_dir, "data_availability.csv"), index=False)
    return tab

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="BAG symptom/cognition association analysis (fMRI teacher -> sMRI student).")
    ap.add_argument("--cases_csv", type=str, default=None)
    ap.add_argument("--hc_csv", type=str, default=None)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--bin_bag_kind", type=str, default="residualized",
                    choices=["residualized", "biascorr", "raw"],
                    help="BAG used for Fig 2 severity-bin means (default residualized; raw is confounded).")
    ap.add_argument("--fig1_ci", type=str, default="fisher", choices=["fisher", "bootstrap"],
                    help="Figure 1 CI: analytic Fisher-z (df n-k-3) or nonparametric bootstrap "
                         "(re-residualized per resample). Stars remain BH Fisher-z either way.")
    ap.add_argument("--fig1_nboot", type=int, default=2000,
                    help="Bootstrap resamples for --fig1_ci bootstrap.")
    ap.add_argument("--fig3_nboot", type=int, default=5000,
                    help="Paired-bootstrap resamples for the Fig 3 directional r(pair1)>r(pair2) tests.")
    ap.add_argument("--fig3_ymin", type=float, default=None,
                    help="Fig 3 broken-axis break point / data-band floor (default FIG4_YFLOOR=0.5, shared across groups).")
    ap.add_argument("--fig3_ymax", type=float, default=None,
                    help="Fig 3 y-axis upper limit (default: auto-fit to the tallest bracket, capped at FIG4_YMAX_CAP=1.12).")
    ap.add_argument("--seed", type=int, default=0, help="Bootstrap RNG seed (reproducibility).")
    ap.add_argument("--demo", action="store_true", help="Run on synthetic data to validate the pipeline.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if not args.cases_csv or not args.hc_csv:
        raise ValueError("Provide --cases_csv and --hc_csv (or use --demo).")
    cases = pd.read_csv(args.cases_csv)
    hc = pd.read_csv(args.hc_csv)

    hc = add_severity_columns(hc)
    cases = add_severity_columns(cases)
    pooled = pd.concat([hc, cases], ignore_index=True)

    print(f"HC N={len(hc)} | Cases N={len(cases)}")

    avail = availability_table(hc, cases, args.out_dir)
    print("\nData availability (complete-case N: outcome + all model BAGs + confounders):")
    print(avail.to_string(index=False))

    figure_matched_symptoms(cases, args.out_dir,
                            ci_method=args.fig1_ci, n_boot=args.fig1_nboot, seed=args.seed)
    figure_cognition(hc, cases, args.out_dir)
    figure_cross_model(hc, cases, args.out_dir,
                       n_boot=args.fig3_nboot, seed=args.seed,
                       ymin=args.fig3_ymin, ymax=args.fig3_ymax)

    print(f"\nWrote 2 figures + stat tables + data_availability.csv to: {args.out_dir}")


if __name__ == "__main__":
    main()
