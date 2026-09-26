#!/usr/bin/env python3
"""RQ3 analysis on the AgentCIBench subset (dataset/subsets/acb_rq3_subset.json), same metrics as RQ1/RQ2.

Graph: normalized WL (h=2, with order); baseline = mean over the 3 within-source pairs, variant =
mean over the 3x3 source-variant pairs; variant -> family mean within task -> task mean.
Consequence: (output channels, disclosed ground-truth items) per execution, as in RQ2 (GPT-4o judge
where available); inconsistency = fraction of differing pairs; delta = S-V minus S-S; task bootstrap CI.
Writes results/rq3_acb/{graph.csv, consequence.csv, coverage.json}.
"""
import sys as _sys, pathlib as _pl  # scripts are grouped by stage; make sibling stages importable
_sys.path[:0] = [str(_d) for _d in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _d.is_dir()]
import csv, itertools, json, random, statistics, sys
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path[:0] = [str(p) for p in sorted((ROOT / "scripts").iterdir()) if p.is_dir()]
from privacy_mt.operation_similarity import OperationGraph, normalized_wl_similarity  # noqa: E402
import analyze_rq2_consequences as A  # noqa: E402

SUB = json.loads((ROOT / "dataset/subsets/acb_rq3_subset.json").read_text())
FAMILY = {"source": "source", "p01": "Tone", "n01": "Tone", "i01": "Tone", "lang_zh": "Language",
          "lang_ru": "Language", "lang_fr": "Language", "lang_es": "Language", "formulation": "Formulation"}
DIRNAME = {"source": "source__source", "formulation": "formulation__formulation"}
CAMPAIGNS = [  # (condition, model, directory)
    ("single", "Gemini 2.5 Flash", "runs/main/gemini-2.5-flash"),
    ("single", "GPT-4.1 Mini", "runs/main/gpt-4.1-mini"),
    ("single", "Claude Haiku 4.5", "runs/main/claude-haiku-4.5"),
    ("single", "GPT-6 Astra", "runs/rq3_acb/gpt-6-astra"),
    ("clarify", "Gemini 2.5 Flash", "runs/rq3_acb/clarify_gemini-2.5-flash"),
    ("clarify", "GPT-4.1 Mini", "runs/rq3_acb/clarify_gpt-4.1-mini"),
    ("clarify", "Claude Haiku 4.5", "runs/rq3_acb/clarify_claude-haiku-4.5"),
]
OUT = ROOT / "results/rq3_acb"
mean = lambda xs: statistics.fmean(xs) if xs else float("nan")


def form_dir(base, task, form):
    if form in DIRNAME:
        return base / task / DIRNAME[form]
    fam = "tone" if form[0] in "pni" and form[1:].isdigit() else "multilingual"
    return base / task / f"{fam}__{form}"


def runs(d):
    out = []
    for r in ("r0", "r1", "r2"):
        p = d / r
        try:
            if json.loads((p / "result.json").read_text()).get("status") == "infrastructure_error":
                continue
        except (OSError, json.JSONDecodeError):
            continue
        out.append(p)
    return out


def boot(vals, n=2000, rng=random.Random(0)):
    ms = sorted(mean([rng.choice(vals) for _ in vals]) for _ in range(n)); return ms[int(.025 * n)], ms[int(.975 * n) - 1]


def load_output(cond):
    """Output similarity (GPT-4o rubric /4) for one condition, with the RQ1 empty-response rule
    (two empty final responses count as identical)."""
    import evaluate_output_similarity_gpt4o as E
    E.CAMPAIGNS = [ROOT / rel for c, _, rel in CAMPAIGNS if c == cond and (ROOT / rel).exists()]
    E.SUBSET = SUB
    jobs = {j["job_id"]: j for j in E.build_jobs()}
    empty = lambda x: not str(x).strip()
    fix = lambda sc, X, Y, pairs: [4 if empty(X[a]) and empty(Y[b]) else v for v, (a, b) in zip(sc, pairs)]
    BP = [(0, 1), (0, 2), (1, 2)]; SV = [(i, j) for i in range(3) for j in range(3)]
    base, var = {}, {}
    path = OUT / f"output_sim_{cond}" / "evaluator_results.jsonl"
    if not path.exists():
        return base, var
    for line in path.read_text().splitlines():
        r = json.loads(line); j = jobs[r["job_id"]]; S = j["source_outputs"]
        if r["kind"] == "baseline":
            base[(r["model_id"], r["task_id"])] = mean([v / 4 for v in fix(r["parsed"]["scores"], S, S, BP)])
        else:
            outs = {v["form"]: v["outputs"] for v in j["variants"]}
            for v in r["parsed"]["variants"]:
                V = outs[v["form"]]
                var[(r["model_id"], r["task_id"], v["form"])] = (
                    mean([x / 4 for x in fix(v["source_variant_scores"], S, V, SV)]),
                    mean([x / 4 for x in fix(v["variant_internal_scores"], V, V, BP)]))
    return base, var


def main():
    A.load_judge()
    outputs = {c: load_output(c) for c in ("single", "clarify")}
    ann = A.load_annotations()
    OUT.mkdir(exist_ok=True)
    graph_rows, cons_rows, coverage = [], [], {}
    for cond, model, rel in CAMPAIGNS:
        base = ROOT / rel
        if not base.exists():
            continue
        g_task = defaultdict(lambda: defaultdict(list)); c_task = defaultdict(lambda: defaultdict(list))
        n_runs = n_judged = 0
        for task in SUB["tasks"]:
            src = runs(form_dir(base, task, "source"))
            if len(src) < 2:
                continue
            sg = [OperationGraph.from_dict(json.loads((p / "operation_graph.json").read_text())) for p in src]
            so = [outcome(ann[task], p) for p in src]
            gss = mean([normalized_wl_similarity(a, b, h=2, include_order=True) for a, b in itertools.combinations(sg, 2)])
            css = mean([int(a != b) for a, b in itertools.combinations(so, 2)])
            g_task["Baseline"][task].append(gss)
            for form in SUB["forms"][1:]:
                var = runs(form_dir(base, task, form))
                if not var:
                    continue
                n_runs += len(var)
                vg = [OperationGraph.from_dict(json.loads((p / "operation_graph.json").read_text())) for p in var]
                vo = [outcome(ann[task], p) for p in var]
                n_judged += sum(str(p.relative_to(ROOT)) in A.JUDGE for p in var)
                g_task[FAMILY[form]][task].append(mean([normalized_wl_similarity(a, b, h=2, include_order=True)
                                                        for a in sg for b in vg]))
                csv_ = mean([int(a != b) for a in so for b in vo])
                c_task[FAMILY[form]][task].append((css, csv_))
        coverage[f"{cond}|{model}"] = {"variant_runs": n_runs, "judged": n_judged, "tasks": len(g_task["Baseline"])}
        row = {"condition": cond, "model": model}
        for fam in ["Baseline", "Tone", "Language", "Formulation"]:
            row[fam] = round(mean([mean(v) for v in g_task[fam].values()]), 3)
        mid = json.loads((base / "manifest.json").read_text())["model"]
        ob, ov = outputs[cond]
        row["out_Baseline"] = round(mean([ob[(mid, t)] for t in SUB["tasks"] if (mid, t) in ob]), 3)
        for fam in ["Tone", "Language", "Formulation"]:
            per = defaultdict(list); pvv = defaultdict(list)
            for (m, t, f), (sv, vv) in ov.items():
                if m == mid and FAMILY.get(f) == fam:
                    per[t].append(sv); pvv[t].append(vv)
            row[f"out_{fam}"] = round(mean([mean(v) for v in per.values()]), 3)
            row[f"out_{fam}_VV"] = round(mean([mean(v) for v in pvv.values()]), 3)
        asked = [json.loads((p / "result.json").read_text()).get("clarification_rounds", 0) > 0
                 for t in SUB["tasks"] for f in SUB["forms"] for p in runs(form_dir(base, t, f))]
        row["ask_rate"] = round(mean(asked), 3) if cond == "clarify" else None
        graph_rows.append(row)
        for fam in ["Tone", "Language", "Formulation"]:
            per = {t: (mean([x[0] for x in v]), mean([x[1] for x in v])) for t, v in c_task[fam].items()}
            d = [b - a for a, b in per.values()]
            if not d:
                continue
            lo, hi = boot(d) if len(d) > 1 else (float("nan"), float("nan"))
            cons_rows.append({"condition": cond, "model": model, "family": fam, "n_tasks": len(d),
                              "within_source": round(mean([a for a, _ in per.values()]), 3),
                              "source_variant": round(mean([b for _, b in per.values()]), 3),
                              "delta": round(mean(d), 3), "se": round(statistics.stdev(d) / len(d) ** .5, 3) if len(d) > 1 else float("nan"),
                              "ci_low": round(lo, 3), "ci_high": round(hi, 3)})
    for name, rows in [("graph.csv", graph_rows), ("consequence.csv", cons_rows)]:
        with (OUT / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (OUT / "coverage.json").write_text(json.dumps(coverage, indent=1))
    for r in graph_rows: print(r)
    for r in cons_rows: print(r)
    print(json.dumps(coverage))


def outcome(a, d):
    """RQ2 ACB consequence: output channels + disclosed ground-truth items (judge when available)."""
    ev = A.read_events(d)
    writes = frozenset(e.get("operation") for e in ev if e["type"] == "disclosure")
    judged = A.JUDGE.get(str(d.relative_to(ROOT)))
    if judged is not None:
        leaked, missing = judged["leaked_items"], judged["missing_items"]
    else:
        disc = A.norm(" \n ".join(str(e.get("payload", "")) for e in ev if e["type"] == "disclosure"))
        leaked = [x for x in a["must_not_share"] if A.contains(disc, x)]
        missing = [x for x in a["must_share"] if not A.contains(disc, x)]
    shared = frozenset({x for x in a["must_share"] if x not in missing} | set(leaked))
    return (writes, shared)


if __name__ == "__main__":
    main()
