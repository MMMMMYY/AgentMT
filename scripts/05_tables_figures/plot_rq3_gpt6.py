#!/usr/bin/env python3
"""RQ3 figure: GPT-6 Astra on the AgentCIBench subset (single-turn); colours match the RQ2 figure."""
import sys as _sys, pathlib as _pl  # scripts are grouped by stage; make sibling stages importable
_sys.path[:0] = [str(_d) for _d in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _d.is_dir()]
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "results/rq3_acb"
M = "GPT-6 Astra"
G = next(r for r in csv.DictReader((D / "graph.csv").open()) if r["condition"] == "single" and r["model"] == M)
C = {r["family"]: r for r in csv.DictReader((D / "consequence.csv").open()) if r["condition"] == "single" and r["model"] == M}
COLS = {"Baseline": "#B8BDC7", "Tone": "#A9CDE6", "Language": "#69A9D0", "Formulation": "#2F6F9F"}  # Figure 2 palette
EDGE = "#4B5563"
CATS = ["Baseline", "Tone", "Language", "Formulation"]
INK = "#111827"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "pdf.fonttype": 42, "ps.fonttype": 42})
fig, axes = plt.subplots(1, 3, figsize=(15.8, 3.0), gridspec_kw={"width_ratios": [1, 1, 1]})
W = 0.58
for ax, key, title in [(axes[0], "", "Graph similarity"), (axes[1], "out_", "Output similarity")]:
    vals = [float(G[key + c]) for c in CATS]
    ax.bar(range(4), vals, W, color=[COLS[c] for c in CATS], edgecolor=EDGE, linewidth=0.8, zorder=2)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.008, f"{v:.3f}", ha="center", va="bottom", fontsize=11)
    ax.set_xticks(range(4)); ax.set_xticklabels(CATS, fontsize=12)
    ax.set_ylim(0.5, 1.05); ax.set_yticks([0.5, 0.75, 1]); ax.set_yticklabels(["0.5", "0.75", "1"])
    ax.set_title(title, fontsize=15, weight="semibold")
axes[0].set_ylabel("Similarity", fontsize=13)
ax = axes[2]
fams = CATS[1:]
ss = float(C["Tone"]["within_source"])
ax.bar(0, ss, W, color=COLS["Baseline"], edgecolor=EDGE, linewidth=0.8, zorder=2)
for i, fam in enumerate(fams, start=1):
    r = C[fam]; sv = float(r["source_variant"])
    ax.bar(i, sv, W, color=COLS[fam], edgecolor=EDGE, linewidth=0.8, zorder=2)
    ax.errorbar(i, sv, yerr=[[sv - (ss + float(r["ci_low"]))], [ss + float(r["ci_high"]) - sv]],
                fmt="none", ecolor=INK, elinewidth=1.1, capsize=3, zorder=4)
ax.set_xticks(range(4)); ax.set_xticklabels(CATS, fontsize=12)
ax.set_ylim(0, 0.5); ax.set_yticks([0, 0.25, 0.5]); ax.set_yticklabels(["0", "0.25", "0.5"])
ax.set_title("Shared information (ACB)", fontsize=15, weight="semibold")
ax.set_ylabel("Inconsistency rate", fontsize=13)
for a in axes:
    a.grid(axis="y", color="#e4e3df", lw=0.8, zorder=0)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
fig.tight_layout(w_pad=2.0)
fig.savefig(D / "fig_rq3_gpt6.pdf"); fig.savefig(D / "fig_rq3_gpt6.png", dpi=200)
print("saved")
