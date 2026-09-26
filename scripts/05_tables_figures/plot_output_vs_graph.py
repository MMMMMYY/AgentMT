#!/usr/bin/env python3
"""Joint scatter of structural (graph) vs output similarity for source-variant pairs.

Main figure: one joint panel per benchmark (models pooled); points are variants
(3x3 source-variant means), coloured by transformation family; marginal
histograms per family on the top/right; Spearman rho in each panel; dashed
line at output similarity 0.75 (rubric >= 3, "same core result").
Appendix figure: the same scatter faceted by model x benchmark.
"""
import sys as _sys, pathlib as _pl  # scripts are grouped by stage; make sibling stages importable
_sys.path[:0] = [str(_d) for _d in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _d.is_dir()]
import csv, random
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results/rq1_output/analysis/variant_level.csv"
OUT = ROOT / "results/rq1_output/analysis"
DATASETS = ["AI Agent Permissions", "AgentCIBench", "TRAJECT-Bench"]
MODELS = ["Gemini 2.5 Flash", "GPT-4.1 Mini", "Claude Haiku 4.5"]
FAMILIES = [("tone", "Tone", "#2a78d6"), ("multilingual", "Language", "#eb6834"),
            ("formulation", "Formulation", "#1baf7a")]   # validated categorical slots 1-3 (all-pairs)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
rng = random.Random(0)
jit = lambda v: v + rng.uniform(-0.008, 0.008)

rows = list(csv.DictReader(SRC.open()))
for r in rows:
    r["x"], r["y"] = float(r["graph_sv"]), float(r["output_sv"])


def _ranks(v):  # average ranks for ties
    v = np.asarray(v, float); order = np.argsort(v, kind="mergesort"); r = np.empty(len(v)); i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a, b):
    return float(np.corrcoef(_ranks(a), _ranks(b))[0, 1]) if len(a) > 2 else float("nan")


def stratified_spearman(sub):
    """Spearman rho with ranks computed within each benchmark (same as analyze_output_similarity.py)."""
    X, Y = [], []
    for ds in DATASETS:
        part = [r for r in sub if r["dataset"] == ds]
        if part:
            n = len(part); rx = _ranks([r["x"] for r in part]); ry = _ranks([r["y"] for r in part])
            X += list((rx - rx.mean()) / n); Y += list((ry - ry.mean()) / n)
    return float(np.corrcoef(X, Y)[0, 1])


def style(ax):
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03)
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    ax.grid(color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.axhline(0.75, color=MUTED, linestyle=(0, (4, 3)), linewidth=0.9)


def scatter(ax, sub, scale=2.2):
    """Count-weighted scatter: variants at (almost) the same coordinates are merged
    into one marker whose area is proportional to their number, so the large mass at
    (1, 1) stays visible. Families are offset slightly so their markers do not hide
    each other."""
    offsets = {"tone": (-0.012, 0.0), "multilingual": (0.0, 0.0), "formulation": (0.012, 0.0)}
    for fam, _, color in FAMILIES:
        cells = {}
        for r in sub:
            if r["family"] == fam:
                key = (round(r["x"] / 0.05) * 0.05, round(r["y"] * 36) / 36)
                cells[key] = cells.get(key, 0) + 1
        dx, dy = offsets[fam]
        ax.scatter([k[0] + dx for k in cells], [k[1] + dy for k in cells],
                   s=[scale * c for c in cells.values()], color=color, alpha=0.55,
                   linewidths=0.4, edgecolors="white", rasterized=True)
    return spearman([r["x"] for r in sub], [r["y"] for r in sub])


plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})
bins = np.linspace(0, 1, 21)

# ---- main: 3 x 3 joint panels, laid out like Figure 2 ------------------------
# rows = models, columns = transformation families, colour = benchmark
DS_COLORS = [("AI Agent Permissions", "#2a78d6"), ("AgentCIBench", "#eb6834"), ("TRAJECT-Bench", "#1baf7a")]
FAM_LABEL = {f: l for f, l, _ in FAMILIES}


def scatter_by(ax, sub, key, palette, scale):
    offsets = [-0.012, 0.0, 0.012]
    for (name, color), dx in zip(palette, offsets):
        cells = {}
        for r in sub:
            if r[key] == name:
                k = (round(r["x"] / 0.05) * 0.05, round(r["y"] * 36) / 36)
                cells[k] = cells.get(k, 0) + 1
        ax.scatter([k[0] + dx for k in cells], [k[1] for k in cells], s=[scale * c for c in cells.values()],
                   color=color, alpha=0.55, linewidths=0.4, edgecolors="white", rasterized=True)
    return stratified_spearman(sub)


fig = plt.figure(figsize=(15.8, 5.0))
outer = fig.add_gridspec(3, 3, wspace=0.08, hspace=0.12, left=0.085, right=0.995, bottom=0.12, top=0.845)
fig.text(0.006, 0.48, "Output similarity", rotation=90, ha="left", va="center", fontsize=14, color=INK)
for i, m in enumerate(MODELS):
    for j, (fam, flabel, _) in enumerate(FAMILIES):
        g = outer[i, j].subgridspec(1, 2, width_ratios=[6, 1], wspace=0.02)
        ax = fig.add_subplot(g[0, 0]); right = fig.add_subplot(g[0, 1], sharey=ax)
        sub = [r for r in rows if r["model"] == m and r["family"] == fam]
        style(ax); ax.tick_params(labelsize=12)
        rho = scatter_by(ax, sub, "dataset", DS_COLORS, scale=4 if fam == "formulation" else 1.3)
        for ds, color in DS_COLORS:
            xs = [r["x"] for r in sub if r["dataset"] == ds]; ys = [r["y"] for r in sub if r["dataset"] == ds]
            if not xs:
                continue
            w = [1 / len(xs)] * len(xs)
            right.hist(ys, bins=bins, weights=w, histtype="step", color=color, linewidth=1.3, orientation="horizontal")
        right.set_xlim(0, 1.0)   # fractions never exceed 1, so nothing is clipped
        right.axis("off")
        ax.text(0.03, 0.05, f"ρ = {rho:.2f}", transform=ax.transAxes, fontsize=13, color=INK,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85))
        if i == 0:
            ax.set_title(flabel, fontsize=18, color=INK, weight="bold", pad=4)
        if j == 0:
            ax.set_ylabel({"Gemini 2.5 Flash": "Gemini\n2.5 Flash", "GPT-4.1 Mini": "GPT-4.1\nMini",
                           "Claude Haiku 4.5": "Claude\nHaiku 4.5"}[m], fontsize=13, color=INK, labelpad=4)
            ax.set_yticks([0, 0.5, 1]); ax.set_yticklabels(["0", "0.5", "1"])
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
        if i == 2:
            ax.set_xlabel("Structural similarity", fontsize=14, color=INK)
        else:
            plt.setp(ax.get_xticklabels(), visible=False)
handles = [Line2D([], [], marker="o", linestyle="None", markersize=9, color=c, label=d) for d, c in DS_COLORS]
handles.append(Line2D([], [], marker="o", linestyle="None", markersize=5, color=MUTED, alpha=0.5,
                      label="marker area ∝ number of variants"))
handles.append(Line2D([], [], color=MUTED, linestyle=(0, (4, 3)), label="Output similarity = 0.75"))
fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, fontsize=13, bbox_to_anchor=(0.5, 1.005),
           handletextpad=0.3, columnspacing=1.2)
fig.savefig(OUT / "output_vs_graph_joint.pdf", dpi=300)
fig.savefig(OUT / "output_vs_graph_joint.png", dpi=200)

# ---- appendix: model x benchmark ---------------------------------------------
fig, axes = plt.subplots(3, 3, figsize=(12, 11), sharex=True, sharey=True)
for i, m in enumerate(MODELS):
    for j, ds in enumerate(DATASETS):
        ax = axes[i, j]; style(ax); rho = scatter(ax, [r for r in rows if r["model"] == m and r["dataset"] == ds], scale=3)
        ax.set_title((ds + "\n" if i == 0 else "") + f"ρ = {rho:.2f}", fontsize=11, color=INK)
        if j == 0: ax.set_ylabel(f"{m}\nOutput similarity", fontsize=11, color=INK)
        if i == 2: ax.set_xlabel("Structural similarity", fontsize=11, color=INK)
fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(OUT / "output_vs_graph_by_model.pdf", dpi=300)
fig.savefig(OUT / "output_vs_graph_by_model.png", dpi=200)
print("saved")
