#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
SCENARIO_COLORS = {
    "S1": "#e69f00",    # Scenario 1 (ANX / GAD-7)
    "S3": "#56b4e9",    # Scenario 3 (AUD / AUDIT)
    "S3RT": "#cc79a7",  # Scenario 3 (reaction time)
}

@dataclass
class ConfoundConfig:
    """Base columns from which the fixed deconfounding design matrix is built.

    Design matrix (columns partialled out of both student BAG and severity):
        age, age^2, age*sex, age^2*sex, <site dummies, collectively>, eTIV, motion
    Note: sex enters ONLY through the two interaction terms (no standalone main
    effect) — this matches the requested specification.
    """

    age_col: str = "age_at_assessment"
    sex_col: str = "sex"
    site_cols: list = field(default_factory=lambda: ["site_1", "site_2", "site_3"])
    etiv_col: str = "EstimatedTotalIntraCranialVol"
    motion_col: str = "MeanHeadMotionrfMRI"

    def base_columns(self) -> list:
        return [self.age_col, self.sex_col, *self.site_cols, self.etiv_col, self.motion_col]

    def n_terms(self) -> int:
        return 5 + len(self.site_cols) + 2

    def build(self, df: pd.DataFrame) -> np.ndarray:
        age = df[self.age_col].to_numpy(float)
        age_c = age - age.mean()
        sex = df[self.sex_col].to_numpy(float)
        cols = [age_c, age_c**2, sex, age_c * sex, age_c**2 * sex]
        for s in self.site_cols:
            cols.append(df[s].to_numpy(float))
        cols.append(df[self.etiv_col].to_numpy(float))
        cols.append(df[self.motion_col].to_numpy(float))
        return np.column_stack(cols)

@dataclass
class ScenarioSpec:
    """Everything needed to compute one scenario's two-lambda comparison."""
    key: str                 # "S1" / "S3" / "S3RT" (used for colour + table)
    label: str               # label for legend / table
    path_l05: str            # CSV for lambda = 0.5
    path_l08: str            # CSV for lambda = 0.8
    severity_col: str        # symptom-severity / outcome column (e.g. gad7_sum, reaction_time)
    diagnosis_col: str       # ICD-10 flag to filter cases (e.g. ANX_icd10); empty/None -> no filter
    bag_col: str             # student BAG column (e.g. student_bag_biascorr)
    id_col: str              # subject id to merge the two lambda files on
    confounds: ConfoundConfig = field(default_factory=ConfoundConfig)
    diagnosis_value: int = 1  # keep rows where diagnosis_col == this (ignored if no diagnosis_col)

    @property
    def use_diagnosis(self) -> bool:
        """Whether to restrict to diagnosis-specific cases. Empty column -> whole sample."""
        return bool(self.diagnosis_col)

# --------------------------------------------------------------------------- #
# Stats helpers
# --------------------------------------------------------------------------- #
def residualize(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Return residuals of y after OLS regression on X (intercept added)."""
    X1 = np.column_stack([np.ones(len(y)), X]) if X.size else np.ones((len(y), 1))
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return y - X1 @ beta

def partial_pearson(a: np.ndarray, b: np.ndarray, C: np.ndarray) -> float:
    """Partial Pearson correlation between a and b, controlling for columns of C."""
    ra = residualize(a, C)
    rb = residualize(b, C)
    return float(np.corrcoef(ra, rb)[0, 1])

def semipartial_pearson(y: np.ndarray, bag: np.ndarray, C: np.ndarray) -> float:
    """Semipartial r: confounder-residualized BAG vs unadjusted outcome."""
    return float(np.corrcoef(y, residualize(bag, C))[0, 1])

def steiger_overlapping_z(r_jk: float, r_jh: float, r_kh: float, n: int):
    """
    Steiger (1980) Z for comparing two OVERLAPPING DEPENDENT correlations that
    share variable j:  r_jk  vs  r_jh  (r_kh = correlation between the two
    non-shared variables k and h). Same formulation used by cocor's
    `steiger1980`. Returns (Z, p_two_sided).

    Here: j = symptom severity, k = student BAG at lambda 0.5, h = student BAG at
    lambda 0.8. A POSITIVE Z means r_jk > r_jh (0.5 stronger); a NEGATIVE Z means
    r_jh > r_jk (0.8 stronger, i.e. transfer increases the association).
    """
    z_jk = np.arctanh(r_jk)
    z_jh = np.arctanh(r_jh)

    r_bar_sq = ((r_jk + r_jh) / 2.0) ** 2
    # asymptotic covariance of the two Fisher-z's (Dunn & Clark 1969)
    num = (
        r_kh * (1.0 - r_jk**2 - r_jh**2)
        - 0.5 * r_jk * r_jh * (1.0 - r_jk**2 - r_jh**2 - r_kh**2)
    )
    denom = (1.0 - r_jk**2) * (1.0 - r_jh**2)
    cov = num / denom

    z = (z_jk - z_jh) * np.sqrt((n - 3.0) / (2.0 - 2.0 * cov))
    p_two = 2.0 * stats.norm.sf(abs(z))
    return float(z), float(p_two)

def one_sided_p_increase(z: float) -> float:
    """
    One-sided p-value for the hypothesis r(lambda=0.8) > r(lambda=0.5).
    With the sign convention above (z_jk - z_jh), that hypothesis makes Z < 0,
    so the one-sided p is the lower-tail probability Phi(Z).
    """
    return float(stats.norm.cdf(z))

def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted q-values."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / (np.arange(m) + 1)
    # enforce monotonicity from the largest p downward
    q_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty_like(q_sorted)
    q[order] = np.clip(q_sorted, 0.0, 1.0)
    return q

# --------------------------------------------------------------------------- #
# Core computation per scenario
# --------------------------------------------------------------------------- #
def compute_scenario(spec: ScenarioSpec) -> dict:
    """Return r05, r08, r_between, n, Steiger Z and one-sided p for a scenario."""
    df05 = pd.read_csv(spec.path_l05)
    df08 = pd.read_csv(spec.path_l08)

    base_conf = spec.confounds.base_columns()
    needed = [spec.id_col, spec.severity_col, spec.bag_col, *base_conf]
    if spec.use_diagnosis:
        needed.append(spec.diagnosis_col)
    for name, df in (("lambda=0.5", df05), ("lambda=0.8", df08)):
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise KeyError(
                f"[{spec.key} {name}] missing columns: {missing}\n"
                f"    available: {list(df.columns)}"
            )

    # restrict to the diagnosis-specific cases in BOTH files (skipped if no filter,
    # e.g. reaction time is a continuous whole-sample outcome)
    if spec.use_diagnosis:
        df05 = df05[df05[spec.diagnosis_col] == spec.diagnosis_value].copy()
        df08 = df08[df08[spec.diagnosis_col] == spec.diagnosis_value].copy()

    # merge on subject id so both correlations use the SAME individuals.
    # severity + confounders are taken from the lambda=0.5 table (same subjects);
    # the BAG columns come from each lambda's own table.
    left = df05[[spec.id_col, spec.severity_col, spec.bag_col, *base_conf]].rename(
        columns={spec.bag_col: "bag_l05"}
    )
    right = df08[[spec.id_col, spec.bag_col]].rename(columns={spec.bag_col: "bag_l08"})
    merged = left.merge(right, on=spec.id_col, how="inner").dropna(
        subset=["bag_l05", "bag_l08", spec.severity_col, *base_conf]
    )

    n = len(merged)
    if n < 10:
        raise ValueError(f"[{spec.key}] only {n} matched cases after merge/dropna — check IDs/filter.")

    sev = merged[spec.severity_col].to_numpy(float)
    bag05 = merged["bag_l05"].to_numpy(float)
    bag08 = merged["bag_l08"].to_numpy(float)
    # fixed deconfounding design matrix: age, age^2, age*sex, age^2*sex, sites, eTIV, motion
    C = spec.confounds.build(merged)

    # partial (residualized) correlations
    r05 = semipartial_pearson(sev, bag05, C)
    r08 = semipartial_pearson(sev, bag08, C)
    # correlation between the two BAGs (residualized) — needed by Steiger
    r_between = float(np.corrcoef(residualize(bag05, C), residualize(bag08, C))[0, 1])

    # df adjustment for partial correlation: effective n = n - k confounders
    n_eff = n - C.shape[1]
    z, _ = steiger_overlapping_z(r05, r08, r_between, n_eff)
    p_one = one_sided_p_increase(z)

    return dict(
        key=spec.key, label=spec.label, n=n, n_eff=n_eff,
        r_l05=r05, r_l08=r08, delta=r08 - r05, r_between=r_between,
        steiger_z=z, p_one_sided=p_one,
    )

# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
def make_plot(results: list, outdir: str, lambdas=(0.5, 0.8)) -> str:
    plt.rcParams.update({
        "font.size": 11,
        "font.family": "sans-serif",
        "axes.linewidth": 0.8,
        "svg.fonttype": "none",
    })

    # --- horizontal layout knobs (edit these to narrow the panel) ---
    FIG_WIDTH = 3.0     # overall panel width in inches (smaller = narrower)
    COLUMN_GAP = 0.5    # distance between the two lambda columns (smaller = closer)
    SIDE_MARGIN = 0.15  # blank space left/right of the columns

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 5.0))
    x = [0.0, COLUMN_GAP]  # two lambda positions

    all_r = [r["r_l05"] for r in results] + [r["r_l08"] for r in results]
    rmin, rmax = min(all_r), max(all_r)
    pad = 0.12 * max(abs(rmin), abs(rmax), 1e-9)
    # anchor baseline at r = 0 when all correlations are positive (paper default);
    # extend below 0 only if some correlation (e.g. reaction time) is negative
    lo = min(0.0, rmin - pad)
    hi = rmax + pad

    handles = []
    for r in results:
        c = SCENARIO_COLORS[r["key"]]
        ys = [r["r_l05"], r["r_l08"]]
        ax.plot(x, ys, color=c, lw=2.0, zorder=3, solid_capstyle="round")
        ax.scatter(x, ys, s=70, color=c, edgecolor="white", linewidth=1.2, zorder=4)
        handles.append(Line2D([0], [0], color=c, lw=2.0, marker="o",
                              markeredgecolor="white", markersize=8, label=r["label"]))

    # legend: colour + scenario name only, no frame
    ax.legend(handles=handles, loc="upper left", frameon=False,
              handlelength=1.4, fontsize=9, borderaxespad=0.2)

    ax.set_xlim(-SIDE_MARGIN, COLUMN_GAP + SIDE_MARGIN)
    ax.set_ylim(lo, hi)
    ax.set_xticks(x)
    ax.set_xticklabels([rf"$\lambda = {lambdas[0]}$", rf"$\lambda = {lambdas[1]}$"])
    ax.set_ylabel("Pearson $r$")

    # full black x- and y-axis lines
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(1.0)
    # keep the horizontal reference line at r = 0 even if the axis extends below it
    if lo < 0.0:
        ax.axhline(0.0, color="black", lw=1.0, zorder=1)
    ax.tick_params(axis="x", colors="black", length=0)
    ax.tick_params(axis="y", colors="black")

    fig.tight_layout()
    png = f"{outdir.rstrip('/')}/fig_parallel_coords_transfer.png"
    #pdf = f"{outdir.rstrip('/')}/fig_parallel_coords_transfer.pdf"
    fig.savefig(png, dpi=600, bbox_inches="tight")
    #fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png

# --------------------------------------------------------------------------- #
# Table
# --------------------------------------------------------------------------- #
def build_table(results: list, outdir: str) -> pd.DataFrame:
    q = bh_fdr(np.array([r["p_one_sided"] for r in results]))
    rows = []
    for r, qi in zip(results, q):
        rows.append({
            "scenario": r["key"],
            "diagnosis": r["label"],
            "n_cases": r["n"],
            "r(lambda=0.5)": round(r["r_l05"], 4),
            "r(lambda=0.8)": round(r["r_l08"], 4),
            "delta_r": round(r["delta"], 4),
            "r(BAG05,BAG08)": round(r["r_between"], 4),
            "steiger_Z": round(r["steiger_z"], 4),
            "p_one_sided": r["p_one_sided"],
            "q_BH_FDR": float(qi),
            "sig(q<.05)": "yes" if qi < 0.05 else "no",
        })
    tab = pd.DataFrame(rows)
    base = outdir.rstrip("/")
    tab.to_csv(f"{base}/transfer_significance_table.csv", index=False)
    return tab

# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # required paths
    p.add_argument("--s1-l05", required=True, help="Scenario 1 (ANX) CSV, lambda=0.5")
    p.add_argument("--s1-l08", required=True, help="Scenario 1 (ANX) CSV, lambda=0.8")
    p.add_argument("--s3-l05", required=True, help="Scenario 3 (AUD) CSV, lambda=0.5")
    p.add_argument("--s3-l08", required=True, help="Scenario 3 (AUD) CSV, lambda=0.8")
    # optional second Scenario-3 example: reaction time (added only if both paths given)
    p.add_argument("--s3rt-l05", help="Scenario 3 reaction-time CSV, lambda=0.5")
    p.add_argument("--s3rt-l08", help="Scenario 3 reaction-time CSV, lambda=0.8")
    # column overrides (defaults follow the paper's conventions)
    p.add_argument("--id-col", default="eid")
    p.add_argument("--bag-col", default="student_bag_biascorr")
    p.add_argument("--s1-severity", default="gad7_sum")
    p.add_argument("--s1-diagnosis", default="ANX_icd10")
    p.add_argument("--s3-severity", default="audit_obs")
    p.add_argument("--s3-diagnosis", default="AUD_icd10")
    # reaction-time example: continuous whole-sample outcome by default (no ICD filter).
    # pass --s3rt-diagnosis to restrict to a diagnosis group instead.
    p.add_argument("--s3rt-severity", default="reaction_time")
    p.add_argument("--s3rt-diagnosis", default="",
                   help="optional ICD-10 flag to filter the reaction-time example; empty = whole sample")
    p.add_argument("--s3rt-label", default="Scenario 3 - RT")
    # deconfounding base columns (design matrix is fixed: age, age^2, age*sex,
    # age^2*sex, site dummies collectively, eTIV, motion)
    p.add_argument("--age-col", default="age_at_assessment")
    p.add_argument("--sex-col", default="sex")
    p.add_argument("--site-cols", nargs="+", default=["site_1", "site_2", "site_3"])
    p.add_argument("--etiv-col", default="EstimatedTotalIntraCranialVol")
    p.add_argument("--motion-col", default="MeanHeadMotionrfMRI")
    p.add_argument("--outdir", default=".")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    conf = ConfoundConfig(age_col=a.age_col, sex_col=a.sex_col, site_cols=list(a.site_cols),
                          etiv_col=a.etiv_col, motion_col=a.motion_col)
    specs = [
        ScenarioSpec("S1", "Scenario 1 - ANX", a.s1_l05, a.s1_l08,
                     a.s1_severity, a.s1_diagnosis, a.bag_col, a.id_col, conf),
        ScenarioSpec("S3", "Scenario 3 - AUD", a.s3_l05, a.s3_l08,
                     a.s3_severity, a.s3_diagnosis, a.bag_col, a.id_col, conf),
    ]
    # optional additional Scenario-3 example (reaction time), same two-lambda structure
    if a.s3rt_l05 and a.s3rt_l08:
        specs.append(
            ScenarioSpec("S3RT", a.s3rt_label, a.s3rt_l05, a.s3rt_l08,
                         a.s3rt_severity, a.s3rt_diagnosis, a.bag_col, a.id_col, conf)
        )
    results = [compute_scenario(s) for s in specs]

    png = make_plot(results, a.outdir)
    tab = build_table(results, a.outdir)

    print("\n" + tab.to_string(index=False))
    print(f"\nFigure : {png}")
    print(f"Table  : {a.outdir.rstrip('/')}/transfer_significance_table.csv")
    return results, tab


if __name__ == "__main__":
    main()