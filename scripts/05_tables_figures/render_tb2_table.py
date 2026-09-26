#!/usr/bin/env python3
"""LaTeX table: coding agents on Terminal-Bench 2.0 (graph + output similarity + pass rate).
Reads results/rq3_coding/{graph.csv, output.csv}; pass rate = verifier reward, averaged
run -> form -> family within task -> task (same aggregation as the similarity columns)."""
import sys as _sys, pathlib as _pl  # scripts are grouped by stage; make sibling stages importable
_sys.path[:0] = [str(_d) for _d in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _d.is_dir()]
import csv, json, statistics, sys
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(p) for p in sorted((ROOT / "scripts").iterdir()) if p.is_dir()]
from analyze_tb2 import AGENTS, FAMILY, collect  # noqa: E402

D = ROOT / "results/rq3_coding"
G = {r["agent"]: r for r in csv.DictReader((D / "graph.csv").open())}
O = {r["agent"]: r for r in csv.DictReader((D / "output.csv").open())}
mean = statistics.fmean
f3 = lambda x: f"{float(x):.3f}"


def reward(t):
    try:
        return float(json.loads((t / "result.json").read_text())["verifier_result"]["rewards"]["reward"])
    except Exception:
        return 0.0  # no verifier result (e.g. agent timeout) counts as a fail


def pass_rates(prefix):
    fam_task = defaultdict(lambda: defaultdict(list))
    for tf, ts in collect(prefix).items():
        task, fam, _ = tf.split("__")
        label = "Baseline" if fam == "source" else FAMILY[fam]
        fam_task[label][task].append(mean([reward(t) for t in ts]))
    return {k: mean([mean(v) for v in d.values()]) for k, d in fam_task.items()}


COLS = ["Baseline", "Tone", "Language", "Formulation"]
out = [r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3pt}",
       r"\caption{Coding agents on Terminal-Bench~2.0 (15 tasks, 8 variants, 3 runs each). Graph and output "
       r"columns follow Table~\ref{tab:rq1}; Pass Rate: fraction of runs passing the task's verifier. "
       r"No source--variant $\Delta$ has a 95\% bootstrap CI excluding 0 for graph similarity.}",
       r"\label{tab:rq3_coding}",
       r"\begin{tabular}{llcccccccccccc}", r"\toprule",
       r"\multirow{2}{*}{Agent} & \multirow{2}{*}{Model} & \multicolumn{4}{c}{Graph Similarity} & \multicolumn{4}{c}{Output Similarity} & "
       r"\multicolumn{4}{c}{Pass Rate} \\",
       r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}\cmidrule(lr){11-14}",
       r"& & Base & Tone & Lang. & Form. & Base & Tone & Lang. & Form. & Src. & Tone & Lang. & Form. \\", r"\midrule"]
for agent, prefix in AGENTS.items():
    p = pass_rates(prefix)
    cells = [f3(G[agent][c]) for c in COLS] + [f3(O[agent][c]) for c in COLS] + [f"{p[c]:.2f}" for c in COLS]
    harness, model = agent[:-1].split(" (")
    out.append(f"{harness} & {model} & " + " & ".join(cells) + r" \\")
out += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
(D / "table_tb2.tex").write_text("\n".join(out) + "\n")
print("\n".join(out))
