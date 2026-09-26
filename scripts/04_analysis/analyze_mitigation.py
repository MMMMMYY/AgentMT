#!/usr/bin/env python3
"""Mitigation (GPT-4o information extraction) vs. no mitigation, on dataset/subsets/mitigation_subset.json.

No mitigation = the main-study runs of the same tasks and prompt forms; mitigation = runs/mitigation.
Per model x benchmark: graph similarity (RQ1: WL h=2 with order; baseline = 3 within-source pairs, variant =
3x3 source-variant pairs, variant -> family -> task), consequence inconsistency delta (RQ2 outcome per
benchmark, S-V minus S-S, task bootstrap CI), and output similarity when the GPT-4o scores exist
(results/mitigation/output_sim_{none,mitigation}). Infrastructure errors are excluded.
Writes results/mitigation/{graph.csv, consequence.csv, output.csv, extraction_convergence.csv}.
"""
import sys as _sys, pathlib as _pl  # scripts are grouped by stage; make sibling stages importable
_sys.path[:0] = [str(_d) for _d in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _d.is_dir()]
import csv, itertools, json, random, statistics, sys
from collections import Counter, defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path[:0] = [str(p) for p in sorted((ROOT / "scripts").iterdir()) if p.is_dir()]
from privacy_mt.operation_similarity import OperationGraph, normalized_wl_similarity  # noqa: E402
import analyze_rq2_consequences as A  # noqa: E402

SUB = json.loads((ROOT / "dataset/subsets/mitigation_subset.json").read_text())
OUT = ROOT / "results/mitigation"
FAMILY = {"p01": "Tone", "n01": "Tone", "i01": "Tone", "lang_zh": "Language", "lang_ru": "Language",
          "lang_fr": "Language", "lang_es": "Language", "formulation": "Formulation"}
FAMS = ["Tone", "Language", "Formulation"]
CAMPAIGNS = {  # model -> {condition: dir}
    "Gemini 2.5 Flash": {"none": "runs/main/gemini-2.5-flash", "mitigation": "runs/mitigation/gemini-2.5-flash"},
    "GPT-4.1 Mini": {"none": "runs/main/gpt-4.1-mini", "mitigation": "runs/mitigation/gpt-4.1-mini"},
    "Claude Haiku 4.5": {"none": "runs/main/claude-haiku-4.5", "mitigation": "runs/mitigation/claude-haiku-4.5"},
}
mean = lambda xs: statistics.fmean(xs) if xs else float("nan")


def form_dir(base, task, form):
    if form == "source":
        return base / task / "source__source"
    if form == "formulation":
        return base / task / "formulation__formulation"
    return base / task / (f"tone__{form}" if form in ("p01", "n01", "i01") else f"multilingual__{form}")


def runs(d):
    out = []
    for r in ("r0", "r1", "r2"):
        p = d / r
        try:
            if json.loads((p / "result.json").read_text()).get("status") == "infrastructure_error":
                continue
        except (OSError, json.JSONDecodeError):
            continue
        if (p / "operation_graph.json").exists():
            out.append(p)
    return out


def outcome(a, d):  # identical to scripts/04_analysis/analyze_rq2_consistency.py
    ev = A.read_events(d)
    if a["dataset"] == "AI Agent Permissions":
        return frozenset((A.norm(e["field"]), e.get("scope")) for e in ev if e["type"] == "permission_request")
    if a["dataset"] == "AgentCIBench":
        writes = frozenset(e.get("operation") for e in ev if e["type"] == "disclosure")
        judged = A.JUDGE.get(str(d.relative_to(ROOT)))
        if judged is not None:
            leaked, missing = judged["leaked_items"], judged["missing_items"]
        else:
            leaked = A.indicators(a, d)["leaked_items"]
            disc = A.norm(" \n ".join(str(e.get("payload", "")) for e in ev if e["type"] == "disclosure"))
            missing = [x for x in a["must_share"] if not A.contains(disc, x)]
        return writes, frozenset({x for x in a["must_share"] if x not in missing} | set(leaked))
    return frozenset(Counter(e.get("tool_name") for e in ev if e["type"] == "tool_invocation").items())


def boot(vals, n=2000, rng=random.Random(0)):
    ms = sorted(mean([rng.choice(vals) for _ in vals]) for _ in range(n)); return ms[int(.025 * n)], ms[int(.975 * n) - 1]


OUTPUT_SOURCES = {  # condition -> (evaluator results, campaign dirs, subset restriction used when scoring)
    "none": ("results/rq1_output/evaluator_results.jsonl",
             [c["none"] for c in CAMPAIGNS.values()], None),          # main-study scores, reused
    "mitigation": ("results/mitigation/output_sim_mitigation/evaluator_results.jsonl",
                   [c["mitigation"] for c in CAMPAIGNS.values()], "dataset/subsets/mitigation_subset_flat.json"),
}


def load_output(cond):
    """Per (model_id, task, form): source-variant and in-batch variant-variant output similarity, plus the
    source repeated-run baseline, with the RQ1 rule that two empty final responses count as identical."""
    import evaluate_output_similarity_gpt4o as E
    path, dirs, subset = OUTPUT_SOURCES[cond]
    if not (ROOT / path).exists():
        return {}, {}
    E.CAMPAIGNS = [ROOT / d for d in dirs]
    E.SUBSET = json.loads((ROOT / subset).read_text()) if subset else None
    jobs = {j["job_id"]: j for j in E.build_jobs()}
    empty = lambda x: not str(x).strip()
    fix = lambda sc, X, Y, pairs: [4 if empty(X[a]) and empty(Y[b]) else v for v, (a, b) in zip(sc, pairs)]
    BP = [(0, 1), (0, 2), (1, 2)]; SVP = [(i, j) for i in range(3) for j in range(3)]
    base, var = {}, {}
    for line in (ROOT / path).read_text().splitlines():
        r = json.loads(line); j = jobs.get(r["job_id"])
        if j is None:
            continue
        S = j["source_outputs"]
        if r["kind"] == "baseline":
            base[(r["model_id"], r["task_id"])] = mean([v / 4 for v in fix(r["parsed"]["scores"], S, S, BP)])
        else:
            outs = {v["form"]: v["outputs"] for v in j["variants"]}
            for v in r["parsed"]["variants"]:
                V = outs[v["form"]]
                var[(r["model_id"], r["task_id"], v["form"])] = (
                    mean([x / 4 for x in fix(v["source_variant_scores"], S, V, SVP)]),
                    mean([x / 4 for x in fix(v["variant_internal_scores"], V, V, BP)]))
    return base, var


def graph(p):
    return OperationGraph.from_dict(json.loads((p / "operation_graph.json").read_text()))


def main():
    A.load_judge(); ann = A.load_annotations(); OUT.mkdir(exist_ok=True)
    grows, crows, missing_judge = [], [], Counter()
    for model, conds in CAMPAIGNS.items():
        for cond, rel in conds.items():
            base = ROOT / rel
            for ds, tasks in SUB["tasks"].items():
                g = defaultdict(dict); c = defaultdict(dict)
                for t in tasks:
                    src = runs(form_dir(base, t, "source"))
                    if len(src) < 2:
                        continue
                    sg = [graph(p) for p in src]; so = [outcome(ann[t], p) for p in src]
                    g["Baseline"][t] = [mean([normalized_wl_similarity(x, y, h=2, include_order=True)
                                              for x, y in itertools.combinations(sg, 2)])]
                    ss = mean([int(x != y) for x, y in itertools.combinations(so, 2)])
                    for form in SUB["forms"][1:]:
                        var = runs(form_dir(base, t, form))
                        if not var:
                            continue
                        if ds == "AgentCIBench":
                            missing_judge[(model, cond)] += sum(str(p.relative_to(ROOT)) not in A.JUDGE for p in var + src)
                        vg = [graph(p) for p in var]; vo = [outcome(ann[t], p) for p in var]
                        g[FAMILY[form]].setdefault(t, []).append(
                            mean([normalized_wl_similarity(x, y, h=2, include_order=True) for x in sg for y in vg]))
                        c[FAMILY[form]].setdefault(t, []).append((ss, mean([int(x != y) for x in so for y in vo])))
                row = {"model": model, "dataset": ds, "condition": cond,
                       "n_tasks": len(g["Baseline"])}
                for fam in ["Baseline"] + FAMS:
                    row[fam] = round(mean([mean(v) for v in g[fam].values()]), 3)
                grows.append(row)
                for fam in FAMS:
                    per = [(mean([a for a, _ in v]), mean([b for _, b in v])) for v in c[fam].values()]
                    d = [b - a for a, b in per]
                    if len(d) < 2:
                        continue
                    lo, hi = boot(d)
                    crows.append({"model": model, "dataset": ds, "condition": cond, "family": fam, "n_tasks": len(d),
                                  "within_source": round(mean([a for a, _ in per]), 3),
                                  "source_variant": round(mean([b for _, b in per]), 3),
                                  "delta": round(mean(d), 3), "se": round(statistics.stdev(d) / len(d) ** .5, 3),
                                  "ci_low": round(lo, 3), "ci_high": round(hi, 3)})
    orows = []
    for cond in ("none", "mitigation"):
        ob, ov = load_output(cond)
        if not ov:
            continue
        for model, conds in CAMPAIGNS.items():
            mid = json.loads((ROOT / conds[cond] / "manifest.json").read_text())["model"]
            for ds, tasks in SUB["tasks"].items():
                row = {"model": model, "dataset": ds, "condition": cond,
                       "Baseline": round(mean([ob[(mid, t)] for t in tasks if (mid, t) in ob]), 3)}
                for fam in FAMS:
                    sv, vv = defaultdict(list), defaultdict(list)
                    for (m, t, f), (a, b) in ov.items():
                        if m == mid and t in tasks and FAMILY.get(f) == fam:
                            sv[t].append(a); vv[t].append(b)
                    row[fam] = round(mean([mean(v) for v in sv.values()]), 3)
                    row[f"{fam}_VV"] = round(mean([mean(v) for v in vv.values()]), 3)
                    row[f"{fam}_gap"] = round(row[f"{fam}_VV"] - row[fam], 3)   # in-batch V-V minus S-V
                orows.append(row)
    if orows:
        with (OUT / "output.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(orows[0])); w.writeheader(); w.writerows(orows)
        for r in orows: print(r)
    for name, rows in [("graph.csv", grows), ("consequence.csv", crows)]:
        with (OUT / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    # how often extraction made a variant's specification identical to the source's
    ex = [json.loads(l) for l in (ROOT / "dataset/mitigation_extraction/extractions.jsonl").open()]
    src = {r["task_id"]: r["extracted"].strip() for r in ex if r["form"] == "source"}
    conv = defaultdict(list)
    for r in ex:
        if r["form"] != "source":
            conv[(r["dataset"], FAMILY[r["form"]])].append(int(r["extracted"].strip() == src[r["task_id"]]))
    with (OUT / "extraction_convergence.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["dataset", "family", "identical_to_source", "n"])
        for (ds, fam), v in sorted(conv.items()):
            w.writerow([ds, fam, round(mean(v), 3), len(v)])
    for r in grows: print(r)
    for r in crows: print(r)
    print("ACB executions without GPT-4o judge (string-match fallback):", dict(missing_judge))


if __name__ == "__main__":
    main()
