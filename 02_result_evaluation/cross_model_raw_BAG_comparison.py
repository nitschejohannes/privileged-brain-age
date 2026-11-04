#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

COL_TEACHER = "teacher_bag_raw"
COL_STUDENT = "student_bag_raw"
COL_BASELINE = "baseline_bag_raw"

# node colors
CLR_TEACHER = "#2a9d8f"
CLR_STUDENT = "#e76f51"
CLR_BASELINE = "#264653"

CLR_EDGE_HL = "#d1495b"   # student--teacher (highlighted) edge
CLR_EDGE = "#9a9a9a"      # other edges

# rows: (label, "hc" | ICD-10 flag column)
ROWS: List[Tuple[str, str]] = [
    ("HC", "HC"),
    ("MDD", "MDD_icd10"),
    ("ANX", "ANX_icd10"),
    ("AUD", "AUD_icd10"),
]

# default scenario column headers (override with --titles)
DEFAULT_TITLES = ["Scenario 1", "Scenario 2", "Scenario 3"]

# label font size shared by the scenario (column) and group (row) annotations
ANNOT_FONTSIZE = 13

MIN_N = 10

# -----------------------------------------------------------------------------
# Stats helpers
# -----------------------------------------------------------------------------
def bh_fdr(pvals) -> np.ndarray:
    """Benjamini-Hochberg q-values; NaNs pass through as NaN."""
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
    qraw = ranked * m / np.arange(1, m + 1)
    qraw = np.minimum.accumulate(qraw[::-1])[::-1]
    qq = np.empty(m)
    qq[order] = np.clip(qraw, 0, 1)
    q[idx] = qq
    return q

def star(q: float) -> str:
    if not np.isfinite(q):
        return ""
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    if q < 0.05:
        return "*"
    return ""

def _r(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])

def bootstrap_st_gt(
    xs: np.ndarray, xt: np.ndarray, xb: np.ndarray, other: str,
    n_boot: int, seed: int,
) -> float:
    """One-sided paired-bootstrap p for H1: r(student,teacher) > r(other).

    `other` is 'BT' (baseline--teacher) or 'SB' (student--baseline). The same
    resampled subjects enter both correlations on each draw, so the test respects
    their covariance. Returns NaN when the frame is too small.
    """
    n = len(xs)
    if n < MIN_N:
        return np.nan
    rng = np.random.default_rng(seed)
    diffs = np.full(n_boot, np.nan)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        s, t, b = xs[idx], xt[idx], xb[idx]
        if min(np.std(s), np.std(t), np.std(b)) == 0:
            continue
        r_st = np.corrcoef(s, t)[0, 1]
        r_other = np.corrcoef(s, b)[0, 1] if other == "SB" else np.corrcoef(b, t)[0, 1]
        diffs[i] = r_st - r_other
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) == 0:
        return np.nan
    return float((np.sum(diffs <= 0) + 1) / (len(diffs) + 1))

# -----------------------------------------------------------------------------
# Geometry
# -----------------------------------------------------------------------------
def _chord(r: float) -> float:
    """Chord distance sqrt(2(1-r)); clamped for numerical safety."""
    return float(np.sqrt(max(2.0 * (1.0 - r), 0.0)))

def embed_triangle(r_st: float, r_bt: float, r_sb: float) -> Dict[str, np.ndarray]:
    """Place teacher/student/baseline so that pairwise distances equal the chord
    distances of the three raw-BAG correlations. Teacher at origin, student on the
    +x axis, baseline in the upper half-plane. Returns centroid-centred coords."""
    d_st = _chord(r_st)       # teacher--student
    d_bt = _chord(r_bt)       # teacher--baseline
    d_sb = _chord(r_sb)       # student--baseline

    T = np.array([0.0, 0.0])
    S = np.array([d_st, 0.0])
    if d_st < 1e-9:
        # teacher and student coincide; drop baseline straight up
        B = np.array([0.0, d_bt])
    else:
        xB = (d_bt ** 2 - d_sb ** 2 + d_st ** 2) / (2.0 * d_st)
        yB = np.sqrt(max(d_bt ** 2 - xB ** 2, 0.0))
        B = np.array([xB, yB])

    pts = {"Teacher": T, "Student": S, "Baseline": B}
    c = np.mean(list(pts.values()), axis=0)
    return {k: v - c for k, v in pts.items()}

# absolute label offsets
R_OFF = 0.11   # r-value label, perpendicular-outward from the edge
S_OFF = 0.20   # significance stars

def _edge_annotations(coords: Dict[str, np.ndarray], r_st, r_bt, r_sb):
    """Return the drawing spec for one cell's three edges plus the (x, y) anchor
    points of every r-label and star, so the same geometry drives both the global
    bounding box and the actual drawing.

    Each edge label sits perpendicular-outward from the edge midpoint (away from
    the triangle centroid), which keeps labels on nearly-parallel adjacent edges
    from colliding. Stars go on the edge the student--teacher correlation is
    compared against.
    """
    T, S, B = coords["Teacher"], coords["Student"], coords["Baseline"]
    centroid = np.mean([T, S, B], axis=0)
    spec = [
        # (pa, pb, r, color, lw, is_highlight, has_star_slot)
        (S, T, r_st, CLR_EDGE_HL, 2.6, True, False),   # highlighted reference edge
        (B, T, r_bt, CLR_EDGE, 1.6, False, True),      # ST > BT stars here
        (S, B, r_sb, CLR_EDGE, 1.6, False, True),      # ST > SB stars here
    ]
    anchors = []  # (x, y) of every text we may draw, for bbox
    drawn = []    # richer per-edge dict for drawing
    for pa, pb, r, color, lw, hl, star_slot in spec:
        mid = (pa + pb) / 2.0
        edir = pb - pa
        norm = np.linalg.norm(edir)
        if norm < 1e-9:
            perp = np.array([0.0, 1.0])
        else:
            perp = np.array([-edir[1], edir[0]]) / norm
            if np.dot(perp, mid - centroid) < 0:   # point outward
                perp = -perp
        r_anchor = mid + perp * R_OFF
        s_anchor = mid + perp * S_OFF
        anchors.append(r_anchor)
        anchors.append(s_anchor)
        drawn.append(dict(pa=pa, pb=pb, r=r, color=color, lw=lw, hl=hl,
                          mid=mid, perp=perp,
                          r_anchor=r_anchor, s_anchor=s_anchor))
    pts = [T, S, B] + anchors
    return drawn, np.array(pts)

# -----------------------------------------------------------------------------
# Per-cell computation
# -----------------------------------------------------------------------------
def _group_frame(cases: pd.DataFrame, hc: pd.DataFrame, flag: str) -> pd.DataFrame:
    """Complete-case frame on the three raw-BAG columns for one group."""
    src = hc if flag == "HC" else cases[cases[flag] == 1] if flag in cases.columns else cases.iloc[0:0]
    need = [COL_TEACHER, COL_STUDENT, COL_BASELINE]
    have = [c for c in need if c in src.columns]
    if len(have) < 3:
        return src.iloc[0:0]
    return src.dropna(subset=need).copy()

def compute_cell(
    cases: pd.DataFrame, hc: pd.DataFrame, flag: str, *, n_boot: int, seed: int
) -> Dict:
    """Correlations, embedding coordinates, per-test p-values and n for one cell."""
    frame = _group_frame(cases, hc, flag)
    n = len(frame)
    out = dict(n=n, r_st=np.nan, r_bt=np.nan, r_sb=np.nan,
               coords=None, p_bt=np.nan, p_sb=np.nan)
    if n < MIN_N:
        return out
    s = frame[COL_STUDENT].to_numpy(float)
    t = frame[COL_TEACHER].to_numpy(float)
    b = frame[COL_BASELINE].to_numpy(float)
    out["r_st"] = _r(s, t)
    out["r_bt"] = _r(b, t)
    out["r_sb"] = _r(s, b)
    if all(np.isfinite([out["r_st"], out["r_bt"], out["r_sb"]])):
        out["coords"] = embed_triangle(out["r_st"], out["r_bt"], out["r_sb"])
    out["p_bt"] = bootstrap_st_gt(s, t, b, "BT", n_boot, seed)
    out["p_sb"] = bootstrap_st_gt(s, t, b, "SB", n_boot, seed + 1)
    return out

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def _draw_cell(ax, cell: Dict, q_comb: float, lims) -> None:
    (xmin, xmax, ymin, ymax) = lims
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    coords = cell.get("coords")
    if coords is None:
        ax.text(0.5, 0.5, "n/a", transform=ax.transAxes,
                ha="center", va="center", color="#999", fontsize=11)
        return

    drawn, _ = _edge_annotations(coords, cell["r_st"], cell["r_bt"], cell["r_sb"])
    stars = star(q_comb)  # teacher-likeness: r_ST > r_BT (FDR-corrected)

    for e in drawn:
        pa, pb = e["pa"], e["pb"]
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=e["color"], lw=e["lw"],
                solid_capstyle="round", zorder=1,
                alpha=0.95 if e["hl"] else 0.7)
        ra = e["r_anchor"]
        ax.text(ra[0], ra[1], f"{e['r']:.2f}", ha="center", va="center",
                fontsize=7.8, color="#333", zorder=4)
        # a single set of stars on the highlighted student--teacher edge,
        # placed just beyond the r-value (same outward side, directly under it)
        if e["hl"] and stars:
            sp = e["mid"] + e["perp"] * (R_OFF + 0.075)
            ax.text(sp[0], sp[1], stars, ha="center", va="center",
                    fontsize=11, color="#000", fontweight="bold", zorder=5)

    for name, clr in [("Teacher", CLR_TEACHER), ("Student", CLR_STUDENT),
                      ("Baseline", CLR_BASELINE)]:
        p = coords[name]
        ax.scatter([p[0]], [p[1]], s=90, color=clr, zorder=3,
                   edgecolors="white", linewidths=1.0)

def build_triangle_grid(
    scenarios: List[Dict], out_path: str, *,
    n_boot: int = 5000, seed: int = 0, titles: Optional[List[str]] = None,
) -> pd.DataFrame:
    """scenarios: list of dicts {cases: DataFrame, hc: DataFrame}. Writes the grid
    to out_path and returns the per-cell stats as a tidy DataFrame."""
    titles = titles or DEFAULT_TITLES[: len(scenarios)]
    n_scen = len(scenarios)

    # ---- pass 1: compute every cell; both directional tests, both FDR-corrected
    # The FIGURE stars the teacher-likeness test r_ST > r_BT. Both this test and
    # the second directional test r_ST > r_SB (one-sided paired bootstraps) are
    # BH-FDR corrected, each within its own family of the twelve group x scenario
    # cells, and both are written to the stats table.
    cells: Dict[Tuple[int, int], Dict] = {}
    p_bt: List[float] = []
    p_sb: List[float] = []
    p_keys: List[Tuple[int, int]] = []
    for ci, sc in enumerate(scenarios):
        for ri, (_, flag) in enumerate(ROWS):
            cell = compute_cell(sc["cases"], sc["hc"], flag,
                                n_boot=n_boot, seed=seed + 100 * ci + 10 * ri)
            cells[(ri, ci)] = cell
            p_bt.append(cell["p_bt"]); p_sb.append(cell["p_sb"])
            p_keys.append((ri, ci))
    q_bt = bh_fdr(p_bt)     # ST > BT family (12 cells) -> figure stars
    q_sb = bh_fdr(p_sb)     # ST > SB family (12 cells) -> stats table
    qmap = {k: q for k, q in zip(p_keys, q_bt)}
    qmap_sb = {k: q for k, q in zip(p_keys, q_sb)}
 
    # ---- common data box across all triangles (nodes + label anchors) --------
    xs, ys = [], []
    for cell in cells.values():
        if cell.get("coords") is not None:
            _, pts = _edge_annotations(cell["coords"], cell["r_st"],
                                       cell["r_bt"], cell["r_sb"])
            xs.extend(pts[:, 0]); ys.extend(pts[:, 1])
    if xs:
        pad = 0.06 * max(max(xs) - min(xs), max(ys) - min(ys))
        xmin, xmax = min(xs) - pad, max(xs) + pad
        ymin, ymax = min(ys) - pad, max(ys) + pad
    else:
        xmin, xmax, ymin, ymax = -1, 1, -1, 1
    lims = (xmin, xmax, ymin, ymax)
    box_ratio = (ymax - ymin) / (xmax - xmin)
 
    # ---- draw (cell aspect matches the data box, so no wasted whitespace) ----
    cell_w = 2.7
    fig, axes = plt.subplots(
        len(ROWS), n_scen,
        figsize=(cell_w * n_scen + 0.7, cell_w * box_ratio * len(ROWS) + 1.0),
        squeeze=False,
    )
    for ri, (rlabel, _) in enumerate(ROWS):
        for ci in range(n_scen):
            ax = axes[ri][ci]
            cell = cells[(ri, ci)]
            _draw_cell(ax, cell, qmap[(ri, ci)], lims)
            if ri == 0:
                ax.set_title(titles[ci], fontsize=ANNOT_FONTSIZE,
                             fontweight="bold", pad=10)
            if ci == 0:
                ax.set_ylabel(rlabel, fontsize=ANNOT_FONTSIZE, fontweight="bold",
                              rotation=0, ha="right", va="center", labelpad=18)
 
    # shared legend (node identity) + a one-line note on the asterisks
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=CLR_TEACHER,
               markeredgecolor="white", markersize=10, label="Teacher"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=CLR_STUDENT,
               markeredgecolor="white", markersize=10, label="Student"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=CLR_BASELINE,
               markeredgecolor="white", markersize=10, label="Baseline"),
        Line2D([0], [0], color=CLR_EDGE_HL, lw=2.6,
               label="edge of interest (Student–Teacher)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=11, bbox_to_anchor=(0.5, -0.012))
 
    fig.tight_layout(rect=(0.02, 0.02, 1, 1))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
 
    # tidy stats out (both tests FDR-corrected, each with its own stars column)
    rows = []
    for (ri, ci), cell in cells.items():
        rows.append(dict(
            scenario=ci + 1, group=ROWS[ri][0], n=cell["n"],
            r_st=cell["r_st"], r_bt=cell["r_bt"], r_sb=cell["r_sb"],
            p_ST_gt_BT=cell["p_bt"], q_ST_gt_BT=qmap[(ri, ci)],
            stars_ST_gt_BT=star(qmap[(ri, ci)]),
            p_ST_gt_SB=cell["p_sb"], q_ST_gt_SB=qmap_sb[(ri, ci)],
            stars_ST_gt_SB=star(qmap_sb[(ri, ci)]),
        ))
    return pd.DataFrame(rows).sort_values(["scenario", "group"]).reset_index(drop=True)

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="4x3 teacher-likeness triangle grid.")
    for i in (1, 2, 3):
        ap.add_argument(f"--scen{i}_cases", type=str, default=None)
        ap.add_argument(f"--scen{i}_hc", type=str, default=None)
    ap.add_argument("--out", type=str, required=True, help="output image path")
    ap.add_argument("--nboot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stats_csv", type=str, default=None,
                    help="optional path to also write the per-cell stats table")
    args = ap.parse_args()

    scenarios = []
    for i in (1, 2, 3):
        cpath = getattr(args, f"scen{i}_cases")
        hpath = getattr(args, f"scen{i}_hc")
        if not cpath or not hpath:
            raise ValueError(
                f"scenario {i}: provide --scen{i}_cases and --scen{i}_hc "
                f"(or use --demo)."
            )
        scenarios.append({
            "cases": pd.read_csv(cpath),
            "hc": pd.read_csv(hpath),
        })

    stats = build_triangle_grid(scenarios, args.out, n_boot=args.nboot, seed=args.seed)
    print(stats.to_string(index=False))
    if args.stats_csv:
        stats.to_csv(args.stats_csv, index=False)
        print(f"\nstats -> {args.stats_csv}")
    print(f"figure -> {args.out}")


if __name__ == "__main__":
    main()