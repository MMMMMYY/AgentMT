"""Aggregate TB2 output-similarity scores (no API calls).

Same rules as analyze_output_similarity.py: scores/4, two empty final messages count as
identical (4), aggregation variant -> family mean within task -> task mean.
Delta = S-V minus S-S baseline, 95% CI from a task-level bootstrap (10k, seed 0).
Writes results/rq3_coding/output.csv
"""
import sys as _sys, pathlib as _pl  # scripts are grouped by stage; make sibling stages importable
_sys.path[:0] = [str(_d) for _d in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _d.is_dir()]
import csv, itertools, json, random, statistics, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.argv = sys.argv[:1]
sys.path[:0] = [str(p) for p in sorted((ROOT / "scripts").iterdir()) if p.is_dir()]
import evaluate_tb2_output as T  # noqa: E402

RES = ROOT / "results/rq3_coding/output_sim/evaluator_results.jsonl"
FAM = {"tone": "Tone", "multilingual": "Language", "formulation": "Formulation"}
mean = lambda xs: statistics.fmean(xs) if xs else float("nan")
empty = lambda s: not str(s).strip()

JOBS = {j["job_id"]: j for j in T.build_jobs()}
base, var, n_fixed = {}, defaultdict(dict), 0
for line in RES.read_text().splitlines():
    r = json.loads(line)
    job = JOBS[r["job_id"]]; S = job["source_outputs"]
    if r["kind"] == "baseline":
        pairs = list(itertools.combinations(range(3), 2)); raw = r["parsed"]["scores"]
        sc = [4 if empty(S[a]) and empty(S[b]) else s for s, (a, b) in zip(raw, pairs)]
        n_fixed += sum(x != y for x, y in zip(sc, raw))
        base[(r["model_id"], r["task_id"])] = mean([s / 4 for s in sc])
        continue
    outs = {v["form"]: v["outputs"] for v in job["variants"]}
    for v in r["parsed"]["variants"]:
        V = outs[v["form"]]
        svp = [(i, j) for i in range(len(S)) for j in range(len(V))]
        vvp = list(itertools.combinations(range(len(V)), 2))
        sv = [4 if empty(S[a]) and empty(V[b]) else s for s, (a, b) in zip(v["source_variant_scores"], svp)]
        vv = [4 if empty(V[a]) and empty(V[b]) else s for s, (a, b) in zip(v["variant_internal_scores"], vvp)]
        n_fixed += sum(x != y for x, y in zip(sv + vv, v["source_variant_scores"] + v["variant_internal_scores"]))
        var[(r["model_id"], r["family"])].setdefault(r["task_id"], []).append(
            (mean([s / 4 for s in sv]), mean([s / 4 for s in vv]) if vv else None))
print("empty-empty pair scores set to 4:", n_fixed)


def boot(d, n=10_000, seed=0):
    rng, keys = random.Random(seed), list(d)
    xs = sorted(mean([d[rng.choice(keys)] for _ in keys]) for _ in range(n))
    return xs[int(0.025 * n)], xs[int(0.975 * n) - 1]


rows = []
for agent, mid in T.AGENT_IDS.items():
    b = {t: v for (m, t), v in base.items() if m == mid}
    row = {"agent": agent, "n_tasks": len(b), "Baseline": round(mean(list(b.values())), 3)}
    for fam, lab in FAM.items():
        per = var[(mid, fam)]
        sv = {t: mean([x[0] for x in l]) for t, l in per.items()}
        vv = {t: mean([x[1] for x in l if x[1] is not None]) for t, l in per.items()}
        d = {t: sv[t] - b[t] for t in sv}
        lo, hi = boot(d)
        row |= {lab: round(mean(list(sv.values())), 3), f"{lab} VV": round(mean(list(vv.values())), 3),
                f"{lab} Δ": round(mean(list(d.values())), 3),
                f"{lab} se": round(statistics.stdev(d.values()) / len(d) ** 0.5, 3),
                f"{lab} CI": f"[{lo:.3f}, {hi:.3f}]", f"{lab} sig": not (lo <= 0 <= hi)}
    rows.append(row)
out = ROOT / "results/rq3_coding/output.csv"
with out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows:
    print(json.dumps(r, ensure_ascii=False))
