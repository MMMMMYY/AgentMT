#!/usr/bin/env python3
"""Compute RQ1 complete-graph similarities with 3/9/3 run pairing.

For each task and variant:
  * source internal: C(3, 2) = 3 pairs;
  * source--variant: 3 x 3 = 9 pairs;
  * variant internal: C(3, 2) = 3 pairs.

Aggregation is deliberately performed within a task before dataset/model
summaries so tasks with more eligible variants do not receive more weight.
"""
from __future__ import annotations
import sys as _sys, pathlib as _pl  # scripts are grouped by stage; make sibling stages importable
_sys.path[:0] = [str(_d) for _d in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _d.is_dir()]

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from privacy_mt.operation_similarity import OperationGraph, normalized_wl_similarity


DEFAULT_CAMPAIGNS = [
    ROOT / "runs/main/gemini-2.5-flash",
    ROOT / "runs/main/gpt-4.1-mini",
    ROOT / "runs/main/claude-haiku-4.5",
]

MODEL_NAMES = {
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "gpt-4.1-mini-2025-04-14": "GPT-4.1 Mini",
    "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
}


def dataset_name(task_id: str) -> str:
    if task_id.startswith("AIP-"):
        return "AI Agent Permissions"
    if task_id.startswith("ACB-"):
        return "AgentCIBench"
    return "TRAJECT-Bench"


def load_graph(path: Path) -> OperationGraph:
    return OperationGraph.from_dict(json.loads(path.read_text(encoding="utf-8")))


def similarity(left: OperationGraph, right: OperationGraph, h: int) -> float:
    return normalized_wl_similarity(left, right, h=h, include_order=True)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values)


def percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot take a percentile of an empty sample")
    position = probability * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def bootstrap_ci(values: list[float], *, key: str, samples: int) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], values[0]
    seed = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        estimates.append(mean(values[rng.randrange(len(values))] for _ in values))
    estimates.sort()
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaigns", nargs="+", type=Path, default=DEFAULT_CAMPAIGNS)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/rq1_graph",
    )
    parser.add_argument("--h", type=int, default=2)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)

    task_metadata = {
        row["taskId"]: row
        for row in json.loads(
            (ROOT / "dataset/environments/fixed_sources.json").read_text(
                encoding="utf-8"
            )
        )["unified"]
    }

    pair_rows: list[dict] = []
    variant_rows: list[dict] = []
    campaign_records = []

    for campaign_arg in args.campaigns:
        campaign = campaign_arg if campaign_arg.is_absolute() else ROOT / campaign_arg
        manifest = json.loads((campaign / "manifest.json").read_text(encoding="utf-8"))
        if manifest["repeats"] != 3:
            raise ValueError(f"{campaign}: expected exactly three repeats")
        model_id = manifest["model"]
        model = MODEL_NAMES.get(model_id, model_id)
        campaign_records.append({"path": str(campaign), "model": model, "model_id": model_id})
        by_task: dict[str, list[dict]] = {}
        for row in manifest["episodes"]:
            by_task.setdefault(row["task_id"], []).append(row)

        for task_id, forms in sorted(by_task.items()):
            source = next(row for row in forms if row["family"] == "source")
            source_graphs = [
                load_graph(
                    campaign
                    / task_id
                    / f"source__{source['form']}"
                    / f"r{repeat}"
                    / "operation_graph.json"
                )
                for repeat in range(3)
            ]
            baseline_scores = []
            for left in range(3):
                for right in range(left + 1, 3):
                    score = similarity(source_graphs[left], source_graphs[right], args.h)
                    baseline_scores.append(score)
                    pair_rows.append(
                        {
                            "model": model,
                            "dataset": dataset_name(task_id),
                            "task_id": task_id,
                            "family": "source",
                            "form": "source",
                            "comparison": "source_internal",
                            "left_repeat": left,
                            "right_repeat": right,
                            "similarity": score,
                            "strict_mt_eligible": True,
                        }
                    )
            baseline_mean = mean(baseline_scores)

            for row in forms:
                if row["family"] == "source":
                    continue
                variant_graphs = [
                    load_graph(
                        campaign
                        / task_id
                        / f"{row['family']}__{row['form']}"
                        / f"r{repeat}"
                        / "operation_graph.json"
                    )
                    for repeat in range(3)
                ]
                cross_scores = []
                for source_repeat in range(3):
                    for variant_repeat in range(3):
                        score = similarity(
                            source_graphs[source_repeat],
                            variant_graphs[variant_repeat],
                            args.h,
                        )
                        cross_scores.append(score)
                        pair_rows.append(
                            {
                                "model": model,
                                "dataset": dataset_name(task_id),
                                "task_id": task_id,
                                "family": row["family"],
                                "form": row["form"],
                                "comparison": "source_variant",
                                "left_repeat": source_repeat,
                                "right_repeat": variant_repeat,
                                "similarity": score,
                                "strict_mt_eligible": row.get("strict_mt_eligible", True),
                            }
                        )
                internal_scores = []
                for left in range(3):
                    for right in range(left + 1, 3):
                        score = similarity(variant_graphs[left], variant_graphs[right], args.h)
                        internal_scores.append(score)
                        pair_rows.append(
                            {
                                "model": model,
                                "dataset": dataset_name(task_id),
                                "task_id": task_id,
                                "family": row["family"],
                                "form": row["form"],
                                "comparison": "variant_internal",
                                "left_repeat": left,
                                "right_repeat": right,
                                "similarity": score,
                                "strict_mt_eligible": row.get("strict_mt_eligible", True),
                            }
                        )
                cross_mean = mean(cross_scores)
                variant_rows.append(
                    {
                        "model": model,
                        "model_id": model_id,
                        "dataset": dataset_name(task_id),
                        "task_id": task_id,
                        "task_category": task_metadata[task_id].get("taskType"),
                        "domain": task_metadata[task_id].get("domain"),
                        "family": row["family"],
                        "form": row["form"],
                        "tone": row.get("tone"),
                        "language": row.get("language"),
                        "subtype": row.get("subtype"),
                        "strict_mt_eligible": row.get("strict_mt_eligible", True),
                        "source_internal_mean": baseline_mean,
                        "source_variant_3x3_mean": cross_mean,
                        "delta_baseline_minus_variant": baseline_mean - cross_mean,
                        "variant_internal_mean": mean(internal_scores),
                        "source_internal_min": min(baseline_scores),
                        "source_internal_max": max(baseline_scores),
                        "source_variant_min": min(cross_scores),
                        "source_variant_max": max(cross_scores),
                        "variant_internal_min": min(internal_scores),
                        "variant_internal_max": max(internal_scores),
                    }
                )

    # Average variants within a family and task before cross-task summaries.
    task_family_rows = []
    for strict_only in (True, False):
        subset_name = "strict" if strict_only else "all"
        groups: dict[tuple, list[dict]] = {}
        for row in variant_rows:
            if strict_only and not row["strict_mt_eligible"]:
                continue
            key = (row["model"], row["dataset"], row["task_id"], row["family"])
            groups.setdefault(key, []).append(row)
        for (model, dataset, task_id, family), values in sorted(groups.items()):
            task_family_rows.append(
                {
                    "subset": subset_name,
                    "model": model,
                    "dataset": dataset,
                    "task_id": task_id,
                    "task_category": values[0]["task_category"],
                    "domain": values[0]["domain"],
                    "family": family,
                    "eligible_variants": len(values),
                    "source_internal_mean": values[0]["source_internal_mean"],
                    "source_variant_3x3_mean": mean(
                        value["source_variant_3x3_mean"] for value in values
                    ),
                    "delta_baseline_minus_variant": mean(
                        value["delta_baseline_minus_variant"] for value in values
                    ),
                    "variant_internal_mean": mean(
                        value["variant_internal_mean"] for value in values
                    ),
                }
            )

    # One table row per dataset/model. Baseline always averages source tasks;
    # each family delta averages its task-level family values.
    baseline_by_group: dict[tuple, dict[str, float]] = {}
    for row in variant_rows:
        key = (row["model"], row["dataset"], row["task_id"])
        baseline_by_group[key] = row["source_internal_mean"]

    table_rows = []
    long_summary_rows = []
    for subset in ("strict", "all"):
        model_datasets = sorted({(row["model"], row["dataset"]) for row in variant_rows})
        for model, dataset in model_datasets:
            baselines = [
                value
                for (candidate_model, candidate_dataset, _), value in baseline_by_group.items()
                if candidate_model == model and candidate_dataset == dataset
            ]
            base_low, base_high = bootstrap_ci(
                baselines,
                key=f"{subset}|{model}|{dataset}|baseline",
                samples=args.bootstrap_samples,
            )
            output_row = {
                "subset": subset,
                "dataset": dataset,
                "model": model,
                "n_source_tasks": len(baselines),
                "baseline_graph_similarity_mean": mean(baselines),
                "baseline_graph_similarity_ci_low": base_low,
                "baseline_graph_similarity_ci_high": base_high,
            }
            for family in ("tone", "multilingual", "formulation"):
                family_rows = [
                    row
                    for row in task_family_rows
                    if row["subset"] == subset
                    and row["model"] == model
                    and row["dataset"] == dataset
                    and row["family"] == family
                ]
                deltas = [row["delta_baseline_minus_variant"] for row in family_rows]
                cross = [row["source_variant_3x3_mean"] for row in family_rows]
                internal = [row["variant_internal_mean"] for row in family_rows]
                low, high = bootstrap_ci(
                    deltas,
                    key=f"{subset}|{model}|{dataset}|{family}|delta",
                    samples=args.bootstrap_samples,
                )
                output_row[f"{family}_n_tasks"] = len(family_rows)
                output_row[f"{family}_source_variant_similarity_mean"] = mean(cross)
                output_row[f"{family}_delta_mean"] = mean(deltas)
                output_row[f"{family}_delta_ci_low"] = low
                output_row[f"{family}_delta_ci_high"] = high
                output_row[f"{family}_variant_internal_mean"] = mean(internal)
                long_summary_rows.append(
                    {
                        "subset": subset,
                        "dataset": dataset,
                        "model": model,
                        "family": family,
                        "n_tasks": len(family_rows),
                        "baseline_graph_similarity_mean": mean(baselines),
                        "source_variant_similarity_mean": mean(cross),
                        "delta_mean": mean(deltas),
                        "delta_ci_low": low,
                        "delta_ci_high": high,
                        "variant_internal_similarity_mean": mean(internal),
                    }
                )
            table_rows.append(output_row)

    write_jsonl(output / "pair_level_scores.jsonl", pair_rows)
    write_csv(output / "variant_level_scores.csv", variant_rows)
    write_jsonl(output / "variant_level_scores.jsonl", variant_rows)
    write_csv(output / "task_family_scores.csv", task_family_rows)
    write_jsonl(output / "task_family_scores.jsonl", task_family_rows)
    write_csv(output / "summary_long.csv", long_summary_rows)
    write_csv(output / "table_strict.csv", [row for row in table_rows if row["subset"] == "strict"])
    write_csv(output / "table_all.csv", [row for row in table_rows if row["subset"] == "all"])
    metadata = {
        "schema": "rq1_complete_operation_graph_3x3_v1",
        "h": args.h,
        "include_order": True,
        "source_internal_pairs_per_task": 3,
        "source_variant_pairs_per_task_variant": 9,
        "variant_internal_pairs_per_task_variant": 3,
        "aggregation": "variant means first; family means within task; task means within dataset/model",
        "delta": "source_internal_mean - source_variant_3x3_mean",
        "bootstrap": {
            "unit": "task",
            "samples": args.bootstrap_samples,
            "interval": "percentile 95%",
        },
        "campaigns": campaign_records,
        "pair_rows": len(pair_rows),
        "variant_rows": len(variant_rows),
        "task_family_rows": len(task_family_rows),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
