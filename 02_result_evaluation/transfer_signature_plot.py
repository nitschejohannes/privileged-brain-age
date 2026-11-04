#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =============================================================================
# CONFIG  -- edit paths here, then run.
# =============================================================================
SCENARIOS: List[dict] = [
    dict(
        label="Scenario 1",
        hc_csv=".../hc_norm_holdout_predictions.csv",
        cases_csv=".../cases_final_predictions.csv",
    ),
    dict(
        label="Scenario 3",
        hc_csv=".../hc_norm_holdout_predictions.csv",
        cases_csv=".../cases_final_predictions.csv",
    ),
]
OUT_DIR = "..."
N_BOOT = 1000    # bootstrap resamples for the signal-recovery CIs
SEED = 0
# =============================================================================
AGE = "age_at_assessment"
SEX = "sex"
SITE_COLS = ["site_1", "site_2", "site_3"]
TIV = "EstimatedTotalIntraCranialVol"
MOTION = "MeanHeadMotionrfMRI"

BAG_TEACHER = "teacher_bag_biascorr"
BAG_STUDENT = "student_bag_biascorr"
BAG_BASELINE = "baseline_bag_biascorr"

COGNITIVE = ["fluid_intelligence", "reaction_time", "pairs_matching",
             "prospective_memory", "hand_grip_strength"]

MATCHED = [
    ("phq9_sum", "MDD_icd10", "PHQ-9 / MDD"),
    ("gad7_sum", "ANX_icd10", "GAD-7 / ANX"),
    ("audit_obs", "AUD_icd10", "AUDIT-C / AUD"),
]

MIN_N = 10

SCEN_COLORS = ["#E69F00", "#56b4e9", "#0072B2", "#7A4FA3", "#009E73"]
ADDED_COLOR = "#D55E00"
COG_COLOR = "#CC79A7"
COG_ALPHA_HIGHLIGHT = 1.0
COG_ALPHA_FADED = 0.45

SYMPTOM_EMPHASIZE = {("Scenario 1", "gad7_sum"), ("Scenario 3", "audit_obs")}
COG_EMPHASIZE = {("Scenario 3", "reaction_time")}
 
# -----------------------------------------------------------------------------
# Residualization
# -----------------------------------------------------------------------------
def _design_matrix(frame: pd.DataFrame) -> np.ndarray:
    n = len(frame)
    cols: List[np.ndarray] = [np.ones(n)]
    if AGE in frame:
        age = frame[AGE].to_numpy(float)
        age_c = age - np.nanmean(age)
        cols += [age_c, age_c ** 2]
        if SEX in frame:
            sex = frame[SEX].to_numpy(float)
            cols += [sex, age_c * sex, age_c ** 2 * sex]
    elif SEX in frame:
        cols.append(frame[SEX].to_numpy(float))
    for s in SITE_COLS:
        if s in frame:
            cols.append(frame[s].to_numpy(float))
    if TIV in frame:
        cols.append(frame[TIV].to_numpy(float))
    if MOTION in frame:
        cols.append(frame[MOTION].to_numpy(float))
    X = np.column_stack(cols)
    keep = [0] + [j for j in range(1, X.shape[1]) if np.nanstd(X[:, j]) > 1e-12]
    return X[:, keep]
 
def n_confounds(frame: pd.DataFrame) -> int:
    return int(_design_matrix(frame).shape[1] - 1)
 
def residualize(y: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
    y = np.asarray(y, float)
    X = _design_matrix(frame)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta
 
# -----------------------------------------------------------------------------
# Correlation + inference helpers
# -----------------------------------------------------------------------------
def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])
 
def fisher_ci(r: float, dof: int, alpha: float = 0.05) -> Tuple[float, float]:
    if not np.isfinite(r) or dof <= 0 or abs(r) >= 1.0:
        return (np.nan, np.nan)
    z, se = np.arctanh(r), 1.0 / np.sqrt(dof)
    zc = norm.ppf(1 - alpha / 2)
    return float(np.tanh(z - zc * se)), float(np.tanh(z + zc * se))
 
def fisher_p(r: float, dof: int) -> float:
    if not np.isfinite(r) or dof <= 0 or abs(r) >= 1.0:
        return np.nan
    return float(2 * norm.sf(abs(np.arctanh(r)) / (1.0 / np.sqrt(dof))))
 
def star(p: float) -> str:
    if not np.isfinite(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
 
# -----------------------------------------------------------------------------
# Frames + per-unit correlations
# -----------------------------------------------------------------------------
def _common_frame(df: pd.DataFrame, outcome: str) -> pd.DataFrame:
    bag_cols = [BAG_TEACHER, BAG_STUDENT, BAG_BASELINE]
    conf = [c for c in ([AGE, SEX, TIV, MOTION] + SITE_COLS) if c in df.columns]
    need = [c for c in ([outcome] + bag_cols + conf) if c in df.columns]
    if not all(c in df.columns for c in ([outcome] + bag_cols)):
        return df.iloc[0:0]
    return df.dropna(subset=need).copy()

def resid_r(frame: pd.DataFrame, bag_col: str, outcome: str) -> float:
    """Semi-partial r: confounder-residualized BAG vs raw outcome."""
    bag = residualize(frame[bag_col].to_numpy(float), frame)
    return _pearson(bag, frame[outcome].to_numpy(float))

@dataclass
class Unit:
    scenario: str
    outcome: str
    label: str
    subgroup: str
    kind: str        # "symptom" or "cognition"
    is_signal: bool

def _iter_units(scen_label: str, hc: pd.DataFrame, cases: pd.DataFrame):
    """Yield (Unit, frame) pairs for every outcome x subgroup in one scenario."""
    for sum_col, icd, lab in MATCHED:
        if sum_col in cases.columns and icd in cases.columns:
            sub = cases[cases[icd] == 1]
            yield Unit(scen_label, sum_col, lab, f"{icd}==1", "symptom", True), _common_frame(sub, sum_col)
    for cog in COGNITIVE:
        if cog in cases.columns:
            yield Unit(scen_label, cog, f"{cog} (cases)", "cases", "cognition", False), _common_frame(cases, cog)

# -----------------------------------------------------------------------------
# PLOT 1: transfer-proportionality
# -----------------------------------------------------------------------------
def build_proportionality(scenarios: List[dict]) -> pd.DataFrame:
    rows: List[Dict] = []
    for si, sc in enumerate(scenarios):
        hc = pd.read_csv(sc["hc_csv"]); cases = pd.read_csv(sc["cases_csv"])
        for unit, frame in _iter_units(sc["label"], hc, cases):
            n = len(frame)
            if n < MIN_N:
                continue
            k = n_confounds(frame)
            rt = resid_r(frame, BAG_TEACHER, unit.outcome)
            rs = resid_r(frame, BAG_STUDENT, unit.outcome)
            rb = resid_r(frame, BAG_BASELINE, unit.outcome)
            if not (np.isfinite(rt) and np.isfinite(rs) and np.isfinite(rb)):
                continue
            rows.append(dict(
                scenario=unit.scenario, scen_idx=si, outcome=unit.outcome,
                label=unit.label, subgroup=unit.subgroup, kind=unit.kind,
                is_signal=unit.is_signal, n=n, k=k,
                r_teacher=rt, r_student=rs, r_baseline=rb,
                teacher_gap=rt - rb, student_gap=rs - rb,
            ))
    return pd.DataFrame(rows)
 
def _split_emphasis(sub: pd.DataFrame, keyset) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a subset into (emphasized, rest) by (scenario, outcome) membership."""
    if sub.empty:
        return sub, sub
    m = sub.apply(lambda r: (r["scenario"], r["outcome"]) in keyset, axis=1)
    return sub[m], sub[~m]
 
def plot_proportionality(df: pd.DataFrame, out_path: str) -> dict:
    plt.rcParams.update({"font.size": 11, "axes.linewidth": 0.8})
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
 
    if df.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        fig.savefig(out_path, dpi=600, bbox_inches="tight"); plt.close(fig)
        return {}
 
    x = df["teacher_gap"].to_numpy(float)
    y = df["student_gap"].to_numpy(float)
    lim = 1.05 * np.nanmax(np.abs(np.concatenate([x, y, [0.05]])))
 
    # reference lines
    ax.axhline(0, color="#9AA0A6", lw=1.0, ls=(0, (5, 4)), zorder=1)
    ax.plot([-lim, lim], [-lim, lim], color="#BDBDBD", lw=1.0, ls=(0, (1, 2)), zorder=1)
    ax.set_aspect("equal", adjustable="box")
    ax.text(lim * 1.06, lim * 0.94, "possible transfer", ha="right", va="bottom",
            fontsize=11, color="#8A8A8A", rotation=45, rotation_mode="anchor")
    ax.text(-lim * 0.55, -0.015, "no transfer", ha="right", va="bottom", fontsize=11, color="#8A8A8A")
 
    # symptom (signal) points: highlight the positive winners, fade the rest
    sym = df[df["is_signal"]]
    sym_emph, sym_rest = _split_emphasis(sym, SYMPTOM_EMPHASIZE)
    for si in sorted(df["scen_idx"].unique()):
        c = SCEN_COLORS[si % len(SCEN_COLORS)]
        d = sym_rest[sym_rest["scen_idx"] == si]
        ax.scatter(d["teacher_gap"], d["student_gap"], s=40, color=c,
                   edgecolor="none", alpha=0.45, zorder=3.5)
    for si in sorted(df["scen_idx"].unique()):
        c = SCEN_COLORS[si % len(SCEN_COLORS)]
        d = sym_emph[sym_emph["scen_idx"] == si]
        ax.scatter(d["teacher_gap"], d["student_gap"], s=95, color=c,
                   edgecolor="white", linewidth=1.2, zorder=4)
 
    # cognition (control) points: highlight the positive reaction_time example
    cog = df[~df["is_signal"]]
    cog_emph, cog_rest = _split_emphasis(cog, COG_EMPHASIZE)
    ax.scatter(cog_rest["teacher_gap"], cog_rest["student_gap"], s=30, color=COG_COLOR,
               edgecolor="none", alpha=COG_ALPHA_FADED, zorder=3)
    ax.scatter(cog_emph["teacher_gap"], cog_emph["student_gap"], s=85, color=COG_COLOR,
               edgecolor="white", linewidth=1.2, alpha=COG_ALPHA_HIGHLIGHT, zorder=4.5)
 
    # labels: only the highlighted positive examples
    for _, r in sym_emph.iterrows():
        ax.annotate(r["label"], (r["teacher_gap"], r["student_gap"]),
                    xytext=(6, 4), textcoords="offset points", fontsize=9,
                    color="#333333", zorder=5)
    for _, r in cog_emph.iterrows():
        ax.annotate(f"reaction time ({r['subgroup']})", (r["teacher_gap"], r["student_gap"]),
                    xytext=(6, 4), textcoords="offset points", fontsize=9,
                    color='black', zorder=5)
 
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("Teacher residual component  ($r_{\\mathrm{teacher}}-r_{\\mathrm{baseline}}$)")
    ax.set_ylabel("Student signal recovered  ($r_{\\mathrm{student}}-r_{\\mathrm{baseline}}$)")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
 
    # legend
    handles = [Line2D([0], [0], marker="o", ls="none", color=SCEN_COLORS[si % len(SCEN_COLORS)],
                      markeredgecolor="white", markersize=9,
                      label=df[df["scen_idx"] == si]["scenario"].iloc[0] + " - symptoms")
               for si in sorted(df["scen_idx"].unique())]
    handles += [
        Line2D([0], [0], marker="o", ls="none", color=COG_COLOR, markeredgecolor="white",
               markersize=9, label="Scenario 3 - cognition"),
        #Line2D([0], [0], marker="o", ls="none", color=COG_COLOR, markeredgecolor="none",
        #       alpha=COG_ALPHA_FADED, markersize=7, label="other cognition"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=11, handletextpad=0.4)
 
    #ax.set_title("Transfer is proportional to available teacher signal", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=600, bbox_inches="tight"); plt.close(fig)
    return dict(n_points=len(df))
 
# -----------------------------------------------------------------------------
# Signal-recovery table (replaces the removed slope-through-origin line)
# -----------------------------------------------------------------------------
def bootstrap_recovery(frame: pd.DataFrame, outcome: str, n_boot: int,
                       rng: np.random.Generator) -> Tuple[float, float]:
    """Percentile CI for recovery = (r_student-r_baseline)/(r_teacher-r_baseline).
    Re-residualizes each resample. Wide when the teacher gap is small."""
    m = len(frame); fr: List[float] = []
    for _ in range(n_boot):
        fb = frame.iloc[rng.integers(0, m, m)]
        rt = resid_r(fb, BAG_TEACHER, outcome)
        rs = resid_r(fb, BAG_STUDENT, outcome)
        rb = resid_r(fb, BAG_BASELINE, outcome)
        tg = rt - rb
        if np.isfinite(tg) and abs(tg) > 1e-9 and np.isfinite(rs) and np.isfinite(rb):
            fr.append((rs - rb) / tg)
    if len(fr) < 20:
        return (np.nan, np.nan)
    return float(np.percentile(fr, 2.5)), float(np.percentile(fr, 97.5))

def build_recovery_table(scenarios: List[dict], n_boot: int, seed: int) -> pd.DataFrame:
    """Recovery fraction per point with a POSITIVE teacher gap. 1.0 = full
    transfer, 0 = none, <0 = student moved opposite. With bootstrap 95% CI."""
    rng = np.random.default_rng(seed)
    rows: List[Dict] = []
    for sc in scenarios:
        hc = pd.read_csv(sc["hc_csv"]); cases = pd.read_csv(sc["cases_csv"])
        for unit, frame in _iter_units(sc["label"], hc, cases):
            n = len(frame)
            if n < MIN_N:
                continue
            rt = resid_r(frame, BAG_TEACHER, unit.outcome)
            rs = resid_r(frame, BAG_STUDENT, unit.outcome)
            rb = resid_r(frame, BAG_BASELINE, unit.outcome)
            tgap, sgap = rt - rb, rs - rb
            if not np.isfinite(tgap) or tgap <= 0:
                continue
            lo, hi = bootstrap_recovery(frame, unit.outcome, n_boot, rng)
            rows.append(dict(
                scenario=unit.scenario, outcome=unit.outcome, label=unit.label,
                subgroup=unit.subgroup, kind=unit.kind, n=n,
                r_teacher=round(rt, 4), r_baseline=round(rb, 4), r_student=round(rs, 4),
                teacher_gap=round(tgap, 4), student_gap=round(sgap, 4),
                recovery_fraction=round(sgap / tgap, 4),
                recovery_ci_lo=round(lo, 4) if np.isfinite(lo) else np.nan,
                recovery_ci_hi=round(hi, 4) if np.isfinite(hi) else np.nan,
                recovered=bool(sgap > 0),
            ))
    tab = pd.DataFrame(rows)
    if not tab.empty:
        tab = tab.sort_values(["scenario", "teacher_gap"], ascending=[True, False])
    return tab

# -----------------------------------------------------------------------------
# PLOT 2: distillation-added-component decomposition
# -----------------------------------------------------------------------------
def added_component(frame: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rs = residualize(frame[BAG_STUDENT].to_numpy(float), frame)
    rb = residualize(frame[BAG_BASELINE].to_numpy(float), frame)
    X = np.column_stack([np.ones(len(rb)), rb])
    beta, *_ = np.linalg.lstsq(X, rs, rcond=None)
    added = rs - X @ beta
    return added, rs, rb
 
def plot_added_component(scenarios: List[dict], out_path: str) -> pd.DataFrame:
    plt.rcParams.update({"font.size": 10.5, "axes.linewidth": 0.8})
    ncols = len(MATCHED)
    nrows = len(scenarios)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.3 * nrows), squeeze=False)
    rows: List[Dict] = []
 
    for si, sc in enumerate(scenarios):
        cases = pd.read_csv(sc["cases_csv"])
        for ci, (sum_col, icd, lab) in enumerate(MATCHED):
            ax = axes[si][ci]
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if sum_col not in cases.columns or icd not in cases.columns:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center", transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([]); continue
            sub = cases[cases[icd] == 1]
            frame = _common_frame(sub, sum_col)
            n = len(frame)
            if n < MIN_N:
                ax.text(0.5, 0.5, f"n={n}<{MIN_N}", ha="center", va="center", transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([]); continue
 
            k = n_confounds(frame)
            sev = frame[sum_col].to_numpy(float)
            added, rs, rb = added_component(frame)
 
            r_added = _pearson(added, sev)
            r_base = _pearson(rb, sev)
            r_stud = _pearson(rs, sev)
            rho = _pearson(rs, rb)
            dof_added = n - k - 1 - 3
            p_added = fisher_p(r_added, dof_added)
            lo, hi = fisher_ci(r_added, dof_added)
 
            is_win = (sc["label"], sum_col) in SYMPTOM_EMPHASIZE   # the highlighted panel
            dot_color = SCEN_COLORS[si % len(SCEN_COLORS)]         # dots coloured by scenario
 
            ax.scatter(sev, added, s=16, color=dot_color, alpha=0.55, edgecolor="none", zorder=2)
            if np.nanstd(sev) > 0:
                b1, b0 = np.polyfit(sev, added, 1)
                xs = np.array([np.nanmin(sev), np.nanmax(sev)])
                ax.plot(xs, b0 + b1 * xs, color="#222222", lw=1.8, zorder=3)
            ax.axhline(0, color="#BDBDBD", lw=0.8, ls=(0, (4, 3)), zorder=1)
 
            ax.set_title(lab, fontsize=10.5, fontweight="bold" if is_win else "normal")
            if ci == 0:
                scen_bold = r"$\mathbf{" + sc["label"].replace(" ", r"\ ") + r"}$"
                ax.set_ylabel(f"{scen_bold}\n\ndistillation-added\nBAG component", fontsize=9.5)
            if si == nrows - 1:
                ax.set_xlabel("symptom severity")
 
            r_sym = r"$\mathbf{r}$" if is_win else "$r$"
            added_line = f"added {r_sym} = {r_added:.3f}{star(p_added)}  (n={n})"
            rest_lines = (f"baseline $r$ = {r_base:.3f}\n"
                          f"student $r$ = {r_stud:.3f}")
            ax.text(0.03, 0.97, added_line, transform=ax.transAxes, va="top", ha="left",
                    fontsize=8.5, color="#333333",
                    fontweight="bold" if is_win else "normal")
            ax.annotate(rest_lines, xy=(0.03, 0.97), xycoords="axes fraction",
                        xytext=(0, -11), textcoords="offset points",
                        va="top", ha="left", fontsize=8.5, color="#333333")
 
            rows.append(dict(scenario=sc["label"], outcome=sum_col, subgroup=f"{icd}==1",
                             n=n, k=k, r_added=r_added, ci_lo=lo, ci_hi=hi,
                             p_added=p_added, r_baseline=r_base, r_student=r_stud,
                             rho=rho,
                             r_student_check=rho * r_base
                                             + np.sqrt(max(0.0, 1.0 - rho ** 2)) * r_added))
 
    fig.tight_layout()
    fig.savefig(out_path, dpi=600, bbox_inches="tight"); plt.close(fig)
    return pd.DataFrame(rows)
 
# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
 
    prop = build_proportionality(SCENARIOS)
    prop.to_csv(os.path.join(OUT_DIR, "transfer_proportionality_stats.csv"), index=False)
    plot_proportionality(prop, os.path.join(OUT_DIR, "plot1_transfer_proportionality.png"))
 
    recovery = build_recovery_table(SCENARIOS, n_boot=N_BOOT, seed=SEED)
    recovery.to_csv(os.path.join(OUT_DIR, "signal_recovery_table.csv"), index=False)
 
    added = plot_added_component(SCENARIOS, os.path.join(OUT_DIR, "plot2_added_component.png"))
    added.to_csv(os.path.join(OUT_DIR, "added_component_stats.csv"), index=False)
 
    if not recovery.empty:
        print("Signal recovery (positive teacher-gap points only):")
        print(recovery.to_string(index=False))
    if not added.empty:
        print("\nDistillation-added-component correlations:")
        print(added.to_string(index=False))
    print(f"\nWrote 2 figures + 3 stat CSVs to: {OUT_DIR}")
 
 
if __name__ == "__main__":
    main()