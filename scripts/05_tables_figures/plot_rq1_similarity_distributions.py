#!/usr/bin/env python3
"""Plot task-level RQ1 full-graph similarity distributions.

Each model/transformation panel compares the same eligible tasks under:
  1. within-source repeated-run similarity; and
  2. source--variant 3x3 similarity.

The script emits nine standalone figures plus one 3x3 overview, in both PDF
and PNG formats. Boxes show the IQR and median; diamonds show means; paired
points/lines show task-level observations.
"""
from __future__ import annotations
import sys as _sys, pathlib as _pl  # scripts are grouped by stage; make sibling stages importable
_sys.path[:0] = [str(_d) for _d in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _d.is_dir()]

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "results/rq1_graph/task_family_scores.csv"
DEFAULT_OUTPUT = ROOT / "results/rq1_graph/figures"

MODELS = ["Gemini 2.5 Flash", "GPT-4.1 Mini", "Claude Haiku 4.5"]
FAMILIES = ["tone", "multilingual", "formulation"]
FAMILY_LABELS = {
    "tone": "Tone",
    "multilingual": "Language",
    "formulation": "Formulation",
}
DATASETS = ["AI Agent Permissions", "AgentCIBench", "TRAJECT-Bench"]
DATASET_LABELS = {
    "AI Agent Permissions": "AI Agent\nPermissions",
    "AgentCIBench": "AgentCIBench",
    "TRAJECT-Bench": "TRAJECT-Bench",
}

BASELINE_COLOR = "#B8BDC7"
VARIANT_COLOR = "#69A9D0"
POINT_BASELINE = "#4B5563"
POINT_VARIANT = "#176B9A"
GRID_COLOR = "#D9DDE3"


def load_rows(path: Path, subset: str = "all") -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["subset"] != subset:
                continue
            rows.append(
                {
                    **row,
                    "source_internal_mean": float(row["source_internal_mean"]),
                    "source_variant_3x3_mean": float(row["source_variant_3x3_mean"]),
                }
            )
    return rows


def panel_rows(rows: list[dict[str, object]], model: str, family: str):
    return [r for r in rows if r["model"] == model and r["family"] == family]


def draw_panel(ax, rows, model: str, family: str, *, compact: bool = False) -> None:
    subset = panel_rows(rows, model, family)
    base_positions = np.array([1.0, 3.2, 5.4])
    offset = 0.34
    box_width = 0.56
    rng = np.random.default_rng(20260923)

    for dataset_index, dataset in enumerate(DATASETS):
        ds_rows = sorted(
            [r for r in subset if r["dataset"] == dataset],
            key=lambda r: str(r["task_id"]),
        )
        baseline = np.array([r["source_internal_mean"] for r in ds_rows], dtype=float)
        transformed = np.array([r["source_variant_3x3_mean"] for r in ds_rows], dtype=float)
        center = base_positions[dataset_index]
        pos_b, pos_v = center - offset, center + offset

        bp = ax.boxplot(
            [baseline, transformed],
            positions=[pos_b, pos_v],
            widths=box_width,
            patch_artist=True,
            showfliers=False,
            whis=1.5,
            medianprops={"color": "#111827", "linewidth": 1.7},
            whiskerprops={"color": "#4B5563", "linewidth": 1.0},
            capprops={"color": "#4B5563", "linewidth": 1.0},
            boxprops={"edgecolor": "#4B5563", "linewidth": 1.0},
        )
        bp["boxes"][0].set_facecolor(BASELINE_COLOR)
        bp["boxes"][1].set_facecolor(VARIANT_COLOR)
        bp["boxes"][0].set_alpha(0.72)
        bp["boxes"][1].set_alpha(0.72)

        jitter = rng.uniform(-0.075, 0.075, size=len(ds_rows))
        for idx, (b_value, v_value) in enumerate(zip(baseline, transformed)):
            ax.plot(
                [pos_b + jitter[idx], pos_v + jitter[idx]],
                [b_value, v_value],
                color="#7C8797",
                alpha=0.18,
                linewidth=0.55,
                zorder=1,
            )
        ax.scatter(
            np.full(len(baseline), pos_b) + jitter,
            baseline,
            s=11 if compact else 15,
            color=POINT_BASELINE,
            alpha=0.42,
            edgecolors="none",
            zorder=2,
        )
        ax.scatter(
            np.full(len(transformed), pos_v) + jitter,
            transformed,
            s=11 if compact else 15,
            color=POINT_VARIANT,
            alpha=0.42,
            edgecolors="none",
            zorder=2,
        )
        ax.scatter(
            [pos_b, pos_v],
            [baseline.mean(), transformed.mean()],
            marker="D",
            s=29 if compact else 38,
            color=[POINT_BASELINE, POINT_VARIANT],
            edgecolors="white",
            linewidths=0.7,
            zorder=4,
        )
        # ax.text(
        #     center,
        #     1.035,
        #     # f"n={len(ds_rows)}",
        #     ha="center",
        #     va="bottom",
        #     fontsize=8 if compact else 9,
        #     color="#4B5563",
        # )

    ax.set_xlim(0.1, 6.3)
    ax.set_ylim(-0.03, 1.09)
    ax.set_xticks(base_positions)
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASETS], fontsize=11 if compact else 12)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#6B7280")
        spine.set_linewidth(0.75)


def save_standalone(rows, output: Path, model: str, family: str) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.1))
    draw_panel(ax, rows, model, family)
    ax.set_title(f"{model}: {FAMILY_LABELS[family]} variation", fontsize=16, weight="semibold")
    ax.set_ylabel("Normalized WL graph similarity", fontsize=13)
    # ax.set_xlabel("Dataset", fontsize=13)
    legend = [
        Patch(facecolor=BASELINE_COLOR, edgecolor="#4B5563", label="Source repeated-run baseline"),
        Patch(facecolor=VARIANT_COLOR, edgecolor="#4B5563", label=f"Source–{FAMILY_LABELS[family].lower()} variant"),
        Line2D([], [], marker="D", linestyle="None", color="#374151", markeredgecolor="white", label="Mean"),
    ]
    ax.legend(handles=legend, loc="lower left", frameon=False, fontsize=11)
    plt.tight_layout()
    stem = f"rq1_{model.lower().replace(' ', '_').replace('.', '')}_{family}"
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_overview(rows, output: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(15.8, 6.3), sharey=True)
    for row_index, model in enumerate(MODELS):
        for col_index, family in enumerate(FAMILIES):
            ax = axes[row_index, col_index]
            draw_panel(ax, rows, model, family, compact=True)
            if row_index == 0:
                ax.set_title(FAMILY_LABELS[family], fontsize=16, weight="semibold")
            if col_index == 0:
                ax.set_ylabel(f"{model}\nGraph similarity", fontsize=13)
            if row_index == 2:
                pass  # dataset axis title removed; tick labels name the datasets
            else:
                ax.set_xticklabels([])
                ax.tick_params(axis="x", length=0)
    legend = [
        Patch(facecolor=BASELINE_COLOR, edgecolor="#4B5563", label="Source repeated-run baseline"),
        Patch(facecolor=VARIANT_COLOR, edgecolor="#4B5563", label="Source–variant"),
        Line2D([], [], marker="D", linestyle="None", color="#374151", markeredgecolor="white", label="Mean"),
    ]
    fig.legend(
        handles=legend,
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=12,
        bbox_to_anchor=(0.5, 1.0),
    )
    # fig.suptitle("Task-level operational similarity distributions", fontsize=16, weight="semibold", y=1.055)
    for ax in axes.flat:
        ax.set_yticks([0.0, 0.5, 1.0])  # fewer ticks keep the compressed rows readable
    plt.tight_layout(rect=(0, 0, 1, 0.94), h_pad=0.3, w_pad=0.6)
    fig.savefig(output / "rq1_similarity_distributions_overview.pdf", bbox_inches="tight")
    fig.savefig(output / "rq1_similarity_distributions_overview.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--subset", choices=["all", "strict"], default="all")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    rows = load_rows(args.input, args.subset)
    for model in MODELS:
        for family in FAMILIES:
            save_standalone(rows, args.output, model, family)
    save_overview(rows, args.output)
    print(f"Wrote 20 files to {args.output}")


if __name__ == "__main__":
    main()
