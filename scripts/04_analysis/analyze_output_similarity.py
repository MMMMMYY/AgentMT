#!/usr/bin/env python3
"""Output similarity (GPT-4o rubric 0-4, normalized /4) aggregated exactly like RQ1,
plus its relation to operation-graph similarity.

Outputs (results/rq1_output/analysis/):
  variant_level.csv   one row per (model, task, family, form): graph + output 3x3 means
  table_means.csv     Table-3-style means: baseline and source-variant output similarity
  quadrants.csv       structural vs output consistency split (hierarchical characterization)
  correlation.json    Spearman rank correlation between graph and output similarity
"""
import sys as _sys, pathlib as _pl  # scripts are grouped by stage; make sibling stages importable
_sys.path[:0] = [str(_d) for _d in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _d.is_dir()]
import csv, json, statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OS = ROOT / "results/rq1_output"
RQ1 = ROOT / "results/rq1_graph/variant_level_scores.jsonl"
OUT = OS / "analysis"; OUT.mkdir(exist_ok=True)
FAM = {"tone": "Tone", "multilingual": "Language", "formulation": "Formulation"}
MODELS = ["Gemini 2.5 Flash", "GPT-4.1 Mini", "Claude Haiku 4.5"]
DATASETS = ["AI Agent Permissions", "AgentCIBench", "TRAJECT-Bench"]
mean = lambda xs: statistics.fmean(xs) if xs else float("nan")

# Deterministic post-processing (no API calls): two empty final responses are identical
# (same "no final message" completion state) and are scored 4, which the rubric leaves unspecified.
import sys
sys.path[:0] = [str(p) for p in sorted((ROOT / "scripts").iterdir()) if p.is_dir()]
import evaluate_output_similarity_gpt4o as E
JOBS = {j["job_id"]: j for j in E.build_jobs()}
BP = [(0, 1), (0, 2), (1, 2)]; SVP = [(i, j) for i in range(3) for j in range(3)]
empty = lambda s: not str(s).strip()
fix = lambda scores, A, B, pairs: [4 if empty(A[a]) and empty(B[b]) else sc for sc, (a, b) in zip(scores, pairs)]
n_fixed = 0

base, var = {}, {}
for line in (OS / "evaluator_results.jsonl").read_text().splitlines():
    r = json.loads(line)
    job = JOBS[r["job_id"]]; S = job["source_outputs"]
    if r["kind"] == "baseline":
        sc = fix(r["parsed"]["scores"], S, S, BP); n_fixed += sum(a != b for a, b in zip(sc, r["parsed"]["scores"]))
        base[(r["model_id"], r["task_id"])] = mean([s / 4 for s in sc])
    else:
        outs = {v["form"]: v["outputs"] for v in job["variants"]}
        for v in r["parsed"]["variants"]:
            V = outs[v["form"]]
            sv = fix(v["source_variant_scores"], S, V, SVP); vv = fix(v["variant_internal_scores"], V, V, BP)
            n_fixed += sum(a != b for a, b in zip(sv + vv, v["source_variant_scores"] + v["variant_internal_scores"]))
            var[(r["model_id"], r["task_id"], r["family"], v["form"])] = (mean([s / 4 for s in sv]), mean([s / 4 for s in vv]))
print("empty-empty pair scores set to 4:", n_fixed)

rows = []
for line in RQ1.read_text().splitlines():
    g = json.loads(line)
    key = (g["model_id"], g["task_id"], g["family"], g["form"])
    if key not in var or (g["model_id"], g["task_id"]) not in base:
        continue  # e.g. a task pending re-judgement
    rows.append({"model": g["model"], "dataset": g["dataset"], "task_id": g["task_id"], "family": g["family"],
                 "form": g["form"], "graph_baseline": g["source_internal_mean"], "graph_sv": g["source_variant_3x3_mean"],
                 "output_baseline": base[(g["model_id"], g["task_id"])], "output_sv": var[key][0],
                 "output_variant_internal": var[key][1]})
with (OUT / "variant_level.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

# RQ1-identical aggregation: variant -> family mean within task -> task mean
agg = defaultdict(lambda: defaultdict(list))
for r in rows:
    agg[(r["dataset"], r["model"], r["family"])][r["task_id"]].append(r)
table = []
for ds in DATASETS:
    for m in MODELS:
        tasks_b = {r["task_id"]: r["output_baseline"] for r in rows if r["dataset"] == ds and r["model"] == m}
        row = {"Dataset": ds, "Model": m, "Baseline": round(mean(list(tasks_b.values())), 3)}
        for fam, label in FAM.items():
            per_task = {t: mean([x["output_sv"] for x in lst]) for t, lst in agg[(ds, m, fam)].items()}
            row[label] = round(mean(list(per_task.values())), 3)
            row[f"{label} VV"] = round(mean([mean([x["output_variant_internal"] for x in lst]) for lst in agg[(ds, m, fam)].values()]), 3)
            row[f"{label} Δ"] = round(mean([tasks_b[t] - v for t, v in per_task.items()]), 3)
        table.append(row)
with (OUT / "table_means.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i]); r = [0.0] * len(xs); i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b); ma, mb = mean(ra), mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else float("nan")


def stratified_spearman(sub):
    """Spearman rho after ranking within each benchmark (avoids pooling across benchmarks with different levels)."""
    X, Y = [], []
    for ds in DATASETS:
        part = [r for r in sub if r["dataset"] == ds]
        if not part:
            continue
        rx, ry = ranks([r["graph_sv"] for r in part]), ranks([r["output_sv"] for r in part])
        mx, my, n = mean(rx), mean(ry), len(part)
        X += [(v - mx) / n for v in rx]; Y += [(v - my) / n for v in ry]
    ma, mb = mean(X), mean(Y)
    num = sum((x - ma) * (y - mb) for x, y in zip(X, Y))
    den = (sum((x - ma) ** 2 for x in X) * sum((y - mb) ** 2 for y in Y)) ** 0.5
    return num / den if den else float("nan")


corr = {"stratified_by_benchmark": {m: {f: stratified_spearman([r for r in rows if r["model"] == m and r["family"] == f]) for f in FAM} for m in MODELS},
        "pooled_all": spearman([r["graph_sv"] for r in rows], [r["output_sv"] for r in rows]), "n": len(rows)}
for ds in DATASETS:
    sub = [r for r in rows if r["dataset"] == ds]
    corr[ds] = spearman([r["graph_sv"] for r in sub], [r["output_sv"] for r in sub])
(OUT / "correlation.json").write_text(json.dumps(corr, indent=2) + "\n")

# Hierarchical characterization: structure consistent iff source-variant graph similarity is
# not below the task's repeated-run baseline minus 0.1; outputs consistent iff output similarity >= 0.75
# mirrored for outputs: consistent iff S-V output similarity is not below the in-context V-V
# similarity minus 0.1 (V-V is scored in the same evaluator call as S-V).  Thresholds are reported, not tuned.
quad = []
for ds in DATASETS:
    for m in MODELS:
        sub = [r for r in rows if r["dataset"] == ds and r["model"] == m]
        c = defaultdict(int)
        for r in sub:
            s = "struct_consistent" if r["graph_sv"] >= r["graph_baseline"] - 0.1 else "struct_inconsistent"
            o = "output_consistent" if r["output_sv"] >= r["output_variant_internal"] - 0.1 else "output_inconsistent"
            c[f"{s}|{o}"] += 1
        quad.append({"Dataset": ds, "Model": m, "n": len(sub), **{k: round(v / len(sub), 3) for k, v in sorted(c.items())}})
with (OUT / "quadrants.csv").open("w", newline="") as f:
    keys = list(dict.fromkeys(k for q in quad for k in q))
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(quad)

for t in table: print(t)
print(json.dumps(corr))
for q in quad: print(q)
