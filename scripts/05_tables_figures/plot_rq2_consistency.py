#!/usr/bin/env python3
"""RQ2 figure: consequence inconsistency. Palette follows Figure 2: grey = source repeated-run baseline
(within-source inconsistency), blue shades = source-variant inconsistency per transformation family."""
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
D = ROOT / "results/rq2_consequence"
rows = {(r["dataset"], r["model"], r["family"]): r for r in csv.DictReader((D / "consistency_summary.csv").open())}
PANELS = [("AI Agent Permissions", "Permission decision (AIP)"), ("AgentCIBench", "Shared information (ACB)"),
          ("TRAJECT-Bench", "Tool invocations (TJB)")]
MODELS = [("Gemini 2.5 Flash", "Gemini 2.5\nFlash"), ("GPT-4.1 Mini", "GPT-4.1\nMini"), ("Claude Haiku 4.5", "Claude\nHaiku 4.5")]
BASELINE, EDGE = "#B8BDC7", "#4B5563"
FAMS = [("Tone", "#A9CDE6"), ("Language", "#69A9D0"), ("Formulation", "#2F6F9F")]
INK = "#111827"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "pdf.fonttype": 42, "ps.fonttype": 42})
fig, axes = plt.subplots(1, 3, figsize=(15.8, 3.0), sharey=True)
W = 0.2
for ax, (ds, title) in zip(axes, PANELS):
    for i, (m, _) in enumerate(MODELS):
        ss = float(rows[(ds, m, "Tone")]["within_source"])
        ax.bar(i - 1.5 * W, ss, W * 0.92, color=BASELINE, edgecolor=EDGE, linewidth=0.8, zorder=2)
        for k, (fam, col) in enumerate(FAMS):
            r = rows[(ds, m, fam)]; x = i + (k - 0.5) * W
            sv = float(r["source_variant"])
            ax.bar(x, sv, W * 0.92, color=col, edgecolor=EDGE, linewidth=0.8, zorder=2)
            lo, hi = ss + float(r["ci_low"]), ss + float(r["ci_high"])
            ax.errorbar(x, sv, yerr=[[sv - lo], [hi - sv]], fmt="none", ecolor=INK, elinewidth=1.0, capsize=2.3, zorder=4)
    ax.set_title(title, fontsize=15, weight="semibold")
    ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l in MODELS], fontsize=12)
    ax.set_ylim(0, 1.05); ax.set_yticks([0, 0.5, 1]); ax.set_yticklabels(["0", "0.5", "1"])
    ax.grid(axis="y", color="#D9DDE3", lw=0.7, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("Inconsistency rate", fontsize=13)
handles = [Patch(facecolor=BASELINE, edgecolor=EDGE, label="Source repeated-run\nbaseline")]
handles += [Patch(facecolor=c, edgecolor=EDGE, label=f"Source–{f.lower()}") for f, c in FAMS]
handles += [Line2D([], [], color=INK, lw=1.0, marker="|", markersize=8, label="95% CI of Δ")]
fig.tight_layout(rect=(0, 0, 0.845, 1), w_pad=1.0)
fig.legend(handles=handles, loc="center left", frameon=False, fontsize=12, bbox_to_anchor=(0.85, 0.5),
           handlelength=1.4, labelspacing=0.55, borderaxespad=0)
fig.savefig(D / "fig_rq2_consistency.pdf"); fig.savefig(D / "fig_rq2_consistency.png", dpi=200)
print("saved")
