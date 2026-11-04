#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import os
import platform
import sys
import warnings

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import sklearn
import torch

import brain_age_sim as B
from brain_age_sim import run, make_cohort, gen_distillation

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Aesthetics (shared with the LUPI figures for a consistent paper)
# ---------------------------------------------------------------------------
COLORS  = {"teacher": "#2a9d8f", "baseline": "#264653", "student": "#e76f51"}
MODELS  = ["teacher", "student", "baseline"]
MLABEL  = {"teacher":  "Teacher ($x^*$, privileged)",
           "baseline": "Baseline ($x$, hard labels)",
           "student":  "Student ($x$, distilled)"}
MSHORT  = {"teacher": "Teacher", "baseline": "Baseline", "student": "Student"}

plt.rcParams.update({
    "font.size": 10.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e6e6e6",
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
})

# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------
def _sig(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "n.s."

def paired_test(diff, alternative="greater"):
    """Paired t-test on a per-split difference vector `diff` against 0."""
    n = len(diff)
    m = diff.mean()
    sd = diff.std(ddof=1)
    se = sd / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, df=n - 1)
    t_stat, p = stats.ttest_1samp(diff, 0.0, alternative=alternative)
    return {"mean": m, "lo": m - tcrit * se, "hi": m + tcrit * se,
            "dz": m / sd if sd > 0 else np.nan, "t": t_stat, "p_raw": p}

def r_tests(summary):
    """Across-split test of the BAG-severity correlation per model.
    Per-split r is Fisher z-transformed and tested against 0 with a two-sided
    one-sample t-test over the independent splits; BH-FDR across the models."""
    p_raw = np.array([stats.ttest_1samp(np.arctanh(summary[m]["r_case"]), 0.0).pvalue
                      for m in MODELS])
    p_fdr = bh_fdr(p_raw)
    return {m: (float(p_raw[i]), float(p_fdr[i])) for i, m in enumerate(MODELS)}

def bh_fdr(pvals):
    pvals = np.asarray(pvals)
    n = len(pvals)
    order = pvals.argsort()
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, n + 1)
    adj = np.minimum(1.0, pvals * n / ranks)
    for i in reversed(range(n - 1)):
        adj[order[i]] = min(adj[order[i]], adj[order[i + 1]])
    return adj

# ---------------------------------------------------------------------------
# Data saving
# ---------------------------------------------------------------------------
def save_datasets(reps, n_train, n_test_hc, n_test_case, datadir):
    os.makedirs(datadir, exist_ok=True)
    for r in range(reps):
        cohorts = {
            "train_hc":  make_cohort(n_train,     "hc",   seed=3 * r + 0),
            "test_hc":   make_cohort(n_test_hc,   "hc",   seed=3 * r + 1),
            "test_case": make_cohort(n_test_case, "case", seed=3 * r + 2),
        }
        for name, c in cohorts.items():
            np.savez_compressed(
                os.path.join(datadir, f"split{r:03d}_{name}.npz"),
                x=c["x"], xs=c["xs"], age=c["age"], bag=c["bag"], sev=c["sev"])
    with open(os.path.join(datadir, "data_readme.txt"), "w") as f:
        f.write("Normative brain-age simulation datasets\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"splits : {reps}\n")
        f.write(f"n_train_hc   : {n_train}   (normative training controls)\n")
        f.write(f"n_test_hc    : {n_test_hc}   (held-out controls)\n")
        f.write(f"n_test_case  : {n_test_case}   (cases; test only)\n\n")
        f.write("File naming: split<NNN>_<cohort>.npz\n")
        f.write("  cohort in {train_hc, test_hc, test_case}\n\n")
        f.write("Array keys\n----------\n")
        f.write("  x    (n, 8)  standard features    -- available at train and test\n")
        f.write("  xs   (n, 1)  privileged feature    -- teacher only\n")
        f.write("  age  (n,)    chronological age     -- regression target\n")
        f.write("  bag  (n,)    latent true brain-age gap = u + v\n")
        f.write("  sev  (n,)    symptom severity (NaN for controls)\n\n")
        f.write("Data-generating process: see docstring of brainage_sim.py.\n")
        f.write("Key constants\n-------------\n")
        for k in ("D", "SIG_U", "SIG_V_HC", "V0", "BETA_SV", "SIG_V_RES",
                  "SEV_HI", "V_SCALE", "SIG_X", "SIG_XS"):
            f.write(f"  {k:<10} = {getattr(B, k)}\n")
    print(f"  wrote {3 * reps} dataset files + readme to {datadir}/")

# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------
def main_figure(splits, summary, path):
    # representative split: student r closest to the median student r
    rr = summary["student"]["r_case"]
    rep = int(np.argmin(np.abs(rr - np.median(rr))))
    S = splits[rep]
    sev = S["sev"]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.6, 5.3))

    # ---- Panel A: BAG separation, controls vs cases, per model ------------
    positions, box_data, box_colors, box_hatch = [], [], [], []
    xt, xtl = [], []
    for gi, m in enumerate(MODELS):
        base = gi * 1.0
        for oi, (grp, key, hatch) in enumerate(
                [("Controls", "bag_hc", ""), ("Cases", "bag_case", "///")]):
            positions.append(base + oi * 0.32 - 0.16)
            box_data.append(S[m][key])
            box_colors.append(COLORS[m])
            box_hatch.append(hatch)
        xt.append(base)
        xtl.append(MSHORT[m])

    bp = axA.boxplot(box_data, positions=positions, widths=0.28,
                     patch_artist=True, showfliers=False,
                     medianprops=dict(color="white", lw=1.4),
                     whiskerprops=dict(color="#555555", lw=1.0),
                     capprops=dict(color="#555555", lw=1.0))
    for patch, col, htch in zip(bp["boxes"], box_colors, box_hatch):
        patch.set_facecolor(col)
        patch.set_alpha(0.45 if htch == "" else 0.9)
        patch.set_edgecolor(col)
        patch.set_hatch(htch)
        patch.set_linewidth(1.0)
    axA.axhline(0, color="#999999", lw=0.9, ls=":")
    axA.set_xticks(xt)
    axA.set_xticklabels(xtl, fontsize=14)
    axA.set_ylabel("Brain-age gap (years)", fontsize=14)
    axA.set_title("a", loc="left",#    Brain-age gap: controls vs cases
                  fontsize=14, fontweight="bold")
    legA = [Patch(facecolor="#8a8a8a", alpha=0.45, label="Controls"),
            Patch(facecolor="#8a8a8a", alpha=0.9, hatch="///", label="Cases")]
    axA.legend(handles=legA, frameon=False, loc="upper left", fontsize=11)

    # annotate case-control separation per model
    ymax = max(np.percentile(S[m]["bag_case"], 95) for m in MODELS)
    #for gi, m in enumerate(MODELS):
    #    sep = S[m]["bag_case"].mean() - S[m]["bag_hc"].mean()
    #    axA.text(gi, ymax + 1.5, f"$\\Delta$={sep:+.1f} y", ha="center",
    #             fontsize=9, color=COLORS[m], fontweight="bold")

    # ---- Panel B: severity as a function of BAG ---------------------------
    # Brain-age convention: the brain-age gap is the predictor (x) and symptom
    # severity the dependent variable (y). Pearson r is direction-invariant, so
    # the reported r is unchanged; the slope is severity-points per year of gap.
    for m in MODELS:
        bagm = S[m]["bag_case"]
        o    = np.argsort(bagm)
        xb, ys = bagm[o], sev[o]
        axB.scatter(xb, ys, s=9, color=COLORS[m], alpha=0.25,
                    edgecolors="none", zorder=1)
        b1, b0 = np.polyfit(xb, ys, 1)
        rmean  = summary[m]["r_case"].mean()
        rsd    = summary[m]["r_case"].std()
        axB.plot(xb, b0 + b1 * xb, color=COLORS[m], lw=2.6, zorder=3,
                 label=rf"{MSHORT[m]}: "#  slope {b1:+.2f} pt/y,  "
                       f"$r={rmean:.2f}\\pm{rsd:.2f}$")
    axB.axvline(0, color="#999999", lw=0.9, ls=":")
    axB.set_xlabel("Brain-age gap in cases (years)", fontsize=14)
    axB.set_ylabel("Simulated symptom severity", fontsize=14)
    axB.set_title("b", loc="left",#   Symptom severity vs brain-age gap
                  fontsize=14, fontweight="bold")
    axB.legend(frameon=False, loc="upper left", fontsize=11)
               #,title="mean $\\pm$ SD over splits", title_fontsize=8.8)

    fig.tight_layout()
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}  (representative split #{rep})")

MM = 1 / 25.4
NATURE_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 8, "axes.titlesize": 7,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 11,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.major.pad": 2, "ytick.major.pad": 2,
    "axes.grid": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
}
 
CROSS_PAIRS = [  # (csv label, model_a, model_b, legend label, tick label)
    ("Student-Teacher",  "student",  "teacher",  "Student–teacher",  "S–T"),
    ("Baseline-Teacher", "baseline", "teacher",  "Baseline–teacher", "B–T"),
    ("Student-Baseline", "student",  "baseline", "Student–baseline", "S–B"),
]  # marker halves = COLORS[model_a] | COLORS[model_b]
CROSS_TESTS = [                                  # (label, greater-pair, lesser-pair)
    ("ST > BT", 0, 1),   # the student's BAG is more teacher-like than the baseline's
    ("ST > SB", 0, 2),   # ... and more teacher-like than it resembles the baseline
]
_CROSS_GROUPS = [  # (csv name, panel title, BAG key)
    ("HC",    "Held-out controls", "bag_hc"),
    ("Cases", "Cases",             "bag_case"),
]
_DOT_C  = "#a6a6a6"   # per-split dots
_LINE_C = "#d0d0d0"   # paired lines
 
 
def _cross_corrs(splits, group_key):
    """Per-split Pearson r for every cross-model BAG pair within one group."""
    out = {label: [] for label, *_ in CROSS_PAIRS}
    for S in splits:
        for label, ma, mb, *_ in CROSS_PAIRS:
            out[label].append(np.corrcoef(S[ma][group_key], S[mb][group_key])[0, 1])
    return {k: np.asarray(v) for k, v in out.items()}
 
 
def _cross_bracket(ax, x1, x2, y, label, *, h, lw=0.5, fs=7):
    xl, xr = sorted((x1, x2))
    ax.plot([xl, xl, xr, xr], [y, y + h, y + h, y], color="black", lw=lw,
            clip_on=False, zorder=5, solid_capstyle="butt")
    ax.text((xl + xr) / 2.0, y + h * 0.6, label, ha="center", va="bottom",
            fontsize=fs, zorder=6)
 
 
def figure_cross_model(splits, path, *, ymin=None, show_pairs=False):
    """Cross-model raw-BAG correlations (computed per split across subjects) for
    held-out controls and cases. Dots: individual splits; marker + whisker:
    mean and 95% CI across splits; brackets: one-sided paired t-tests across
    splits (ST>BT, ST>SB), BH-FDR over the whole family. show_pairs=True links
    the three correlations of each split with a thin line. Writes PNG + PDF and a
    *_stats.csv beside it; returns (per-cell rows, test rows)."""
    corr = {g: _cross_corrs(splits, key) for g, _, key in _CROSS_GROUPS}
    n = len(splits)
    tcrit = stats.t.ppf(0.975, df=n - 1)
 
    # --- directional paired tests (FDR over the whole family) ----------------
    test_rows, pvals = [], []
    for gname, *_ in _CROSS_GROUPS:
        for tlabel, hi, lo in CROSS_TESTS:
            res = paired_test(corr[gname][CROSS_PAIRS[hi][0]]
                              - corr[gname][CROSS_PAIRS[lo][0]], "greater")
            test_rows.append(dict(group=gname, test=tlabel, mean=res["mean"],
                                  dz=res["dz"], t=res["t"], p_raw=res["p_raw"]))
            pvals.append(res["p_raw"])
    for row, qq in zip(test_rows, bh_fdr(np.array(pvals))):
        row["q"] = float(qq); row["sig"] = _sig(float(qq))
 
    # --- per-cell summaries ----------------------------------------------------
    stat_rows = []
    for gname, *_ in _CROSS_GROUPS:
        for label, *_ in CROSS_PAIRS:
            r = corr[gname][label]
            m = float(r.mean()); sd = float(r.std(ddof=1)); h = tcrit * sd / np.sqrt(n)
            stat_rows.append(dict(group=gname, pair=label, r_mean=m, r_sd=sd,
                                  ci_lo=m - h, ci_hi=m + h, n_splits=n))
 
    # --- plot --------------------------------------------------------------------
    all_r = np.concatenate([v for g in corr.values() for v in g.values()])
    span = all_r.max() - all_r.min()
    br_h, br_gap = 0.025 * span, 0.10 * span      # bracket tick height / tier gap
    jit = np.random.default_rng(0).uniform(-0.09, 0.09, size=n)
    xpts, xmean = -0.12, 0.20                     # dot cloud left, summary right
 
    with plt.rc_context(NATURE_RC):
        fig, axes = plt.subplots(1, 2, figsize=(89 * MM, 64 * MM), sharey=True)
        tops = []
        for j, (ax, (gname, title, _)) in enumerate(zip(axes, _CROSS_GROUPS)):
            R = np.vstack([corr[gname][lab] for lab, *_ in CROSS_PAIRS])  # (3, n)
            X = np.arange(3)[:, None] + xpts + jit[None, :]
            if show_pairs:
                ax.plot(X, R, color=_LINE_C, lw=0.3, zorder=1)
            ax.scatter(X.ravel(), R.ravel(), s=3.5, color=_DOT_C, lw=0, zorder=2)
 
            for i, (_, ma, mb, *_) in enumerate(CROSS_PAIRS):
                r = R[i]; m = r.mean(); h = tcrit * r.std(ddof=1) / np.sqrt(n)
                ax.errorbar(i + xmean, m, yerr=h, fmt="none", ecolor="black",
                            elinewidth=0.6, capsize=1.8, capthick=0.5, zorder=3)
                ax.plot(i + xmean, m, marker="o", ms=6, fillstyle="left",
                        markerfacecolor=COLORS[ma], markerfacecoloralt=COLORS[mb],
                        markeredgecolor="black", markeredgewidth=0.4, zorder=4)
 
            # significance brackets, stacked above the data of the pairs involved
            tier = 0
            for tr in (t for t in test_rows if t["group"] == gname):
                if tr["sig"] == "n.s.":
                    continue
                _, hi, lo = next(t for t in CROSS_TESTS if t[0] == tr["test"])
                y = max(R[hi].max(), R[lo].max()) + 0.05 * span + tier * br_gap
                _cross_bracket(ax, hi + (xpts + xmean) / 2, lo + (xpts + xmean) / 2, y, tr["sig"], h=br_h)
                tier += 1
            tops.append(R.max() + 0.05 * span + max(tier, 1) * br_gap)
 
            # ticks centred on each pair, abbreviated; the legend spells them out
            ax.set_xticks(np.arange(3) + (xpts + xmean) / 2)
            ax.set_xticklabels([p[4] for p in CROSS_PAIRS])
            ax.set_xlim(-0.5, 2.5)
            ax.set_title(title, pad=3)
            if j > 0:
                ax.spines["left"].set_visible(False)
                ax.tick_params(axis="y", length=0)
 
        lo = (all_r.min() - 0.05 * span) if ymin is None else ymin
        axes[0].set_ylim(lo, max(tops) + 0.03 * span)
        axes[0].set_ylabel("Cross-model BAG correlation, $r$", fontsize=9)
 
        # legend: one two-tone marker per pair, matching the means in the panels
        handles = [Line2D([], [], ls="none", marker="o", ms=6, fillstyle="left",
                          markerfacecolor=COLORS[ma], markerfacecoloralt=COLORS[mb],
                          markeredgecolor="black", markeredgewidth=0.4,
                          label=f"{short}: {lab.lower()}")
                   for _, ma, mb, lab, short in CROSS_PAIRS]
        fig.tight_layout(pad=0.4, w_pad=0.8, rect=(0, 0, 1, 0.90))
        fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
                   bbox_to_anchor=(0.5, 1.0), handletextpad=0.3,
                   columnspacing=0.8, fontsize=6.5)
        fig.savefig(path, dpi=600, bbox_inches="tight")
        fig.savefig(os.path.splitext(path)[0] + ".pdf", bbox_inches="tight")
        plt.close(fig)
 
    # --- CSV -------------------------------------------------------------------
    csv_p = os.path.splitext(path)[0] + "_stats.csv"
    with open(csv_p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "pair", "r_mean", "r_sd", "ci_lo", "ci_hi", "n_splits"])
        for r in stat_rows:
            w.writerow([r["group"], r["pair"], f"{r['r_mean']:.4f}",
                        f"{r['r_sd']:.4f}", f"{r['ci_lo']:.4f}",
                        f"{r['ci_hi']:.4f}", r["n_splits"]])
        w.writerow([])
        w.writerow(["group", "test", "mean_diff", "dz", "t",
                    "p_raw", "q_fdr", "sig"])
        for r in test_rows:
            w.writerow([r["group"], r["test"], f"{r['mean']:.4f}",
                        f"{r['dz']:.4f}", f"{r['t']:.4f}", f"{r['p_raw']:.3e}",
                        f"{r['q']:.3e}", r["sig"]])
    print(f"  wrote {path}  (+ .pdf, {os.path.basename(csv_p)})")
    return stat_rows, test_rows

# ---------------------------------------------------------------------------
# Lambda-sweep figure
# ---------------------------------------------------------------------------
def lambda_sweep(reps, n_train, n_test_hc, n_test_case, lam_grid):
    r_mean, r_sd, sep_mean, sep_sd = [], [], [], []
    for lam in lam_grid:
        _, summ = run(reps=reps, n_train=n_train, n_test_hc=n_test_hc,
                      n_test_case=n_test_case, lam=lam)
        st = summ["student"]
        r_mean.append(st["r_case"].mean()); r_sd.append(st["r_case"].std())
        sep = st["mean_bag_case"] - st["mean_bag_hc"]
        sep_mean.append(sep.mean()); sep_sd.append(sep.std())
    # teacher / baseline references (lambda-independent)
    _, ref = run(reps=reps, n_train=n_train, n_test_hc=n_test_hc,
                 n_test_case=n_test_case, lam=0.0)
    refs = {m: (ref[m]["r_case"].mean(),
                (ref[m]["mean_bag_case"] - ref[m]["mean_bag_hc"]).mean())
            for m in ("teacher", "baseline")}
    return (np.array(lam_grid), np.array(r_mean), np.array(r_sd),
            np.array(sep_mean), np.array(sep_sd), refs)

def lambda_figure(sweep, path):
    lam, r_m, r_s, sep_m, sep_s, refs = sweep
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.8))

    for ax, ym, ys, ylab, ttl, refkey in [
        (ax1, r_m, r_s, "BAG$-$severity correlation $r$",
         "a", 0),#   Association transferred vs distillation weight
        (ax2, sep_m, sep_s, "Case$-$control BAG separation (years)",
         "b", 1)]:#   Case elevation recovered vs distillation weight
        ax.axhline(refs["teacher"][refkey], color=COLORS["teacher"], lw=1.8,
                   ls="--", label="Teacher (upper anchor)")
        ax.axhline(refs["baseline"][refkey], color=COLORS["baseline"], lw=1.8,
                   ls="--", label="Baseline ($\\lambda=0$)")
        ax.plot(lam, ym, "-o", color=COLORS["student"], lw=2.4, ms=6,
                label="Student", zorder=3)
        ax.fill_between(lam, ym - ys, ym + ys, color=COLORS["student"],
                        alpha=0.18, zorder=2)
        ax.set_xlabel("Distillation weight $\\lambda$", fontsize=14)
        ax.set_ylabel(ylab, fontsize=14)
        ax.set_title(ttl, loc="left", fontsize=14, fontweight="bold", pad=8)
    handles, labels = ax1.get_legend_handles_labels()
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               fontsize=11, bbox_to_anchor=(0.5, 1.0))
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")

# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def summary_rows(summary):
    ptab = r_tests(summary)
    rows = []
    for m in MODELS:
        s = summary[m]
        sep = s["mean_bag_case"] - s["mean_bag_hc"]
        rows.append({
            "model": MSHORT[m],
            "r": s["r_case"].mean(),   "r_sd": s["r_case"].std(),
            "r_p_raw": ptab[m][0],     "r_p_fdr": ptab[m][1],
            "slope": s["slope_case"].mean(), "slope_sd": s["slope_case"].std(),
            "bag_hc": s["mean_bag_hc"].mean(),
            "bag_case": s["mean_bag_case"].mean(),
            "sep": sep.mean(), "sep_sd": sep.std(),
        })
    return rows

def test_rows(summary, splits):
    rb, rs = (summary[m]["r_case"] for m in ("baseline", "student"))
    sep_s = summary["student"]["mean_bag_case"] - summary["student"]["mean_bag_hc"]

    # individual-level agreement with the teacher's case BAGs (per split)
    agr_base = np.array([np.corrcoef(S["baseline"]["bag_case"],
                                     S["teacher"]["bag_case"])[0, 1] for S in splits])
    agr_stud = np.array([np.corrcoef(S["student"]["bag_case"],
                                     S["teacher"]["bag_case"])[0, 1] for S in splits])

    tests = [
        ("Student's BAG-severity r > baseline's",
         paired_test(rs - rb, "greater"), "r units"),
        ("Student's case BAGs agree with teacher's more than baseline's do",
         paired_test(agr_stud - agr_base, "greater"), "r units"),
        ("Student separates cases from controls (BAG sep > 0)",
         paired_test(sep_s, "greater"), "years"),
    ]
    p_fdr = bh_fdr(np.array([t[1]["p_raw"] for t in tests]))
    rows = []
    for i, (name, res, unit) in enumerate(tests):
        rows.append({
            "hypothesis": name, "unit": unit,
            "mean": res["mean"], "lo": res["lo"], "hi": res["hi"],
            "dz": res["dz"], "t": res["t"],
            "p_fdr": float(p_fdr[i]), "sig": _sig(float(p_fdr[i])),
        })
    return rows

def write_summary_table(rows, outdir):
    csv_p = os.path.join(outdir, "brainage_results.csv")
    with open(csv_p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.3e}" if k.startswith("r_p_")
                            else f"{v:.4f}" if isinstance(v, float) else v)
                        for k, v in r.items()})
    md = ["| Model | r(BAG, severity) | Slope (pt/y) | Mean BAG controls | "
          "Mean BAG cases | Case$-$control sep. |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['model']} | {r['r']:.3f} ± {r['r_sd']:.3f} "
                  f"| {r['slope']:+.3f} ± {r['slope_sd']:.3f} "
                  f"| {r['bag_hc']:+.2f} | {r['bag_case']:+.2f} "
                  f"| {r['sep']:+.2f} ± {r['sep_sd']:.2f} |")
    md_txt = "\n".join(md)
    with open(os.path.join(outdir, "brainage_results.md"), "w") as f:
        f.write(md_txt + "\n")
    tex = [r"\begin{table}[t]", r"\centering", r"\small",
           r"\begin{tabular}{lrrrrr}", r"\toprule",
           r"Model & $r$(BAG,\,sev.) & Slope (pt/y) & BAG$_{\text{HC}}$ & "
           r"BAG$_{\text{case}}$ & Case$-$HC sep. \\", r"\midrule"]
    for r in rows:
        tex.append(f"{r['model']} & ${r['r']:.3f}\\pm{r['r_sd']:.3f}$ & "
                   f"${r['slope']:+.3f}\\pm{r['slope_sd']:.3f}$ & "
                   f"${r['bag_hc']:+.2f}$ & ${r['bag_case']:+.2f}$ & "
                   f"${r['sep']:+.2f}\\pm{r['sep_sd']:.2f}$ \\\\")
    tex += [r"\bottomrule", r"\end{tabular}",
            r"\caption{Normative brain-age distillation on synthetic data. "
            r"Mean $\pm$ SD over splits. The privileged teacher shows a strong "
            r"BAG--severity coupling and clear case elevation; the deployable "
            r"baseline ($x$ only, chronological-age labels) recovers neither; "
            r"the distilled student recovers a substantial fraction of both "
            r"while, like the baseline, using only $x$ at deployment.}",
            r"\label{tab:brainage}", r"\end{table}"]
    with open(os.path.join(outdir, "brainage_results.tex"), "w") as f:
        f.write("\n".join(tex) + "\n")
    print(f"  wrote {csv_p} (+ .md, .tex)")
    return md_txt

def write_test_table(rows, outdir):
    csv_p = os.path.join(outdir, "brainage_tests.csv")
    with open(csv_p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.3e}" if k == "p_fdr"
                            else f"{v:.4f}" if isinstance(v, float) else v)
                        for k, v in r.items()})
    md = ["| Hypothesis (paired across splits) | Mean effect [95% CI] | "
          "$d_z$ | $p_{FDR}$ | Sig. |",
          "|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['hypothesis']} "
                  f"| {r['mean']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] {r['unit']} "
                  f"| {r['dz']:.2f} | {r['p_fdr']:.2e} | {r['sig']} |")
    md_txt = "\n".join(md)
    with open(os.path.join(outdir, "brainage_tests.md"), "w") as f:
        f.write(md_txt + "\n")
    tex = [r"\begin{table}[t]", r"\centering", r"\small",
           r"\begin{tabular}{lrrrl}", r"\toprule",
           r"Hypothesis (paired across splits) & Effect [95\% CI] & $d_z$ & "
           r"$p_{\text{FDR}}$ & \\", r"\midrule"]
    for r in rows:
        sig = r"\text{n.s.}" if r["sig"] == "n.s." else r["sig"]
        tex.append(f"{r['hypothesis']} & "
                   f"${r['mean']:+.3f}\\,[{r['lo']:+.3f}, {r['hi']:+.3f}]$ & "
                   f"${r['dz']:.2f}$ & ${r['p_fdr']:.1e}$ & ${sig}$ \\\\")
    tex += [r"\bottomrule", r"\end{tabular}",
            r"\caption{Paired hypothesis tests across independent splits, "
            r"one-sided $t$-tests with Benjamini--Hochberg FDR correction. "
            r"$d_z$ is Cohen's $d$ for paired samples. "
            r"$^*p_{\text{FDR}}<.05$, $^{**}<.01$, $^{***}<.001$.}",
            r"\label{tab:brainage_tests}", r"\end{table}"]
    with open(os.path.join(outdir, "brainage_tests.tex"), "w") as f:
        f.write("\n".join(tex) + "\n")
    print(f"  wrote {csv_p} (+ .md, .tex)")
    return md_txt

def write_lambda_csv(sweep, outdir):
    lam, r_m, r_s, sep_m, sep_s, refs = sweep
    p = os.path.join(outdir, "brainage_lambda.csv")
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lambda", "student_r_mean", "student_r_sd",
                    "student_sep_mean", "student_sep_sd"])
        for i in range(len(lam)):
            w.writerow([f"{lam[i]:.2f}", f"{r_m[i]:.4f}", f"{r_s[i]:.4f}",
                        f"{sep_m[i]:.4f}", f"{sep_s[i]:.4f}"])
        w.writerow([])
        w.writerow(["teacher_r", f"{refs['teacher'][0]:.4f}",
                    "teacher_sep", f"{refs['teacher'][1]:.4f}"])
        w.writerow(["baseline_r", f"{refs['baseline'][0]:.4f}",
                    "baseline_sep", f"{refs['baseline'][1]:.4f}"])
    print(f"  wrote {p}")

def write_metadata(args, outdir):
    p = os.path.join(outdir, "run_metadata.txt")
    with open(p, "w") as f:
        f.write("Normative brain-age distillation -- run metadata\n")
        f.write("=" * 50 + "\n\n")
        f.write("Environment\n-----------\n")
        f.write(f"  python       : {sys.version.split()[0]} ({platform.platform()})\n")
        f.write(f"  numpy        : {np.__version__}\n")
        f.write(f"  scipy        : {stats.scipy.__version__ if hasattr(stats,'scipy') else 'N/A'}\n")
        f.write(f"  scikit-learn : {sklearn.__version__}\n")
        f.write(f"  torch        : {torch.__version__}\n")
        f.write(f"  matplotlib   : {matplotlib.__version__}\n\n")
        f.write("Settings\n--------\n")
        f.write(f"  reps         : {args.reps} (independent cohorts / splits)\n")
        f.write(f"  n_train_hc   : {args.n_train}\n")
        f.write(f"  n_test_hc    : {args.n_test_hc}\n")
        f.write(f"  n_test_case  : {args.n_test_case}\n")
        f.write(f"  lambda       : {args.lam} (headline student; fixed, not tuned)\n")
        f.write("  tuning       : none; all estimators use library defaults\n")
        f.write("  teacher      : MLPRegressor on x* (privileged), kept at test\n")
        f.write("  student MLP  : 1 hidden layer (32), ReLU, Adam lr=0.01, 400 epochs\n\n")
        f.write("DGP constants\n-------------\n")
        for k in ("AGE_LO", "AGE_HI", "D", "SIG_U", "SIG_V_HC", "V0", "BETA_SV",
                  "SIG_V_RES", "SEV_HI", "V_SCALE", "SIG_X", "SIG_XS"):
            f.write(f"  {k:<10} = {getattr(B, k)}\n")
    print(f"  wrote {p}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps",        type=int, default=50)
    ap.add_argument("--n-train",     type=int, default=300)
    ap.add_argument("--n-test-hc",   type=int, default=500)
    ap.add_argument("--n-test-case", type=int, default=500)
    ap.add_argument("--lam",         type=float, default=0.8)
    ap.add_argument("--outdir",      default="brainage_outputs")
    ap.add_argument("--datadir",     default="brainage_data")
    ap.add_argument("--skip-data",   action="store_true")
    ap.add_argument("--skip-lambda", action="store_true")
    args = ap.parse_args()

    outdir  = os.path.abspath(args.outdir)
    datadir = os.path.abspath(args.datadir)
    os.makedirs(outdir, exist_ok=True)

    print("=" * 62)
    print("Normative brain-age modelling with privileged information")
    print("=" * 62)
    print(f"  results -> {outdir}")
    print(f"  data    -> {datadir}")
    print(f"  reps={args.reps}  n_train={args.n_train}  "
          f"n_test_hc={args.n_test_hc}  n_test_case={args.n_test_case}  "
          f"lam={args.lam}")
    print("=" * 62)

    common = dict(n_train=args.n_train, n_test_hc=args.n_test_hc,
                  n_test_case=args.n_test_case)

    if not args.skip_data:
        print(f"\n[0/4] Saving simulated datasets")
        save_datasets(args.reps, args.n_train, args.n_test_hc,
                      args.n_test_case, datadir)

    print(f"\n[1/4] Main analysis ({args.reps} splits, lam={args.lam})")
    splits, summary = run(reps=args.reps, lam=args.lam, **common)
    for m in MODELS:
        s = summary[m]
        print(f"  {MSHORT[m]:<9} r={s['r_case'].mean():+.3f}±{s['r_case'].std():.3f}  "
              f"slope={s['slope_case'].mean():+.3f}  "
              f"BAG_hc={s['mean_bag_hc'].mean():+.2f}  "
              f"BAG_case={s['mean_bag_case'].mean():+.2f}")

    srows = summary_rows(summary)
    md1 = write_summary_table(srows, outdir)
    trows = test_rows(summary, splits)
    md2 = write_test_table(trows, outdir)
    main_figure(splits, summary, os.path.join(outdir, "brainage_main.png"))
    figure_cross_model(splits, os.path.join(outdir, "brainage_cross_model.png"))

    if not args.skip_lambda:
        print("\n[2/4] Lambda sweep (transfer vs distillation weight)")
        lam_grid = [0.0, 0.25, 0.5, 0.75, 1.0]
        sweep = lambda_sweep(args.reps, args.n_train, args.n_test_hc,
                             args.n_test_case, lam_grid)
        write_lambda_csv(sweep, outdir)
        lambda_figure(sweep, os.path.join(outdir, "brainage_lambda.png"))
    else:
        print("\n[2/4] Lambda sweep skipped")

    print("\n[3/4] Metadata")
    write_metadata(args, outdir)

    print("\n" + "=" * 62)
    print("MAIN RESULTS")
    print("=" * 62)
    print(md1)
    print("\nHYPOTHESIS TESTS")
    print(md2)
    print("=" * 62)
    print(f"Results -> {outdir}")
    print("=" * 62)


if __name__ == "__main__":
    main()