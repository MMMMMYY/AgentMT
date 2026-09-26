#!/usr/bin/env python3
"""LaTeX table: single-turn vs clarification (AgentCIBench RQ3 subset), from results/rq3_acb."""
import sys as _sys, pathlib as _pl  # scripts are grouped by stage; make sibling stages importable
_sys.path[:0] = [str(_d) for _d in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _d.is_dir()]
import csv
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "results/rq3_acb"
G = {(r["condition"], r["model"]): r for r in csv.DictReader((D / "graph.csv").open())}
C = {(r["condition"], r["model"], r["family"]): r for r in csv.DictReader((D / "consequence.csv").open())}
MODELS = ["Gemini 2.5 Flash", "GPT-4.1 Mini", "Claude Haiku 4.5"]
FAMS = ["Tone", "Language", "Formulation"]
f3 = lambda x: f"{float(x):.3f}"

def cons(r):
    c = rf"{float(r['delta']):+.2f}\,$\pm$\,{float(r['se']):.2f}"
    return rf"\textbf{{{c}}}" if float(r["ci_low"]) > 0 or float(r["ci_high"]) < 0 else c

out = [r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3pt}",
       r"\caption{Single-turn execution versus execution with clarification on the AgentCIBench subset "
       r"(15 tasks, 8 variants, 3 runs each). With clarification, the agent may ask the user up to three "
       r"questions; a GPT-4o user simulator answers only from the prompt the agent received. Graph and output "
       r"columns follow Table~\ref{tab:rq1}; consequence columns report $\Delta$ (source--variant minus "
       r"within-source inconsistency, mean $\pm$ s.e.; bold: 95\% bootstrap CI excludes 0). Ask: fraction of "
       r"executions in which the agent asked at least one question.}",
       r"\label{tab:rq3_clarify}",
       r"\begin{tabular}{llcccccccccccc}", r"\toprule",
       r"\multirow{2}{*}{Model} & \multirow{2}{*}{Setting} & \multicolumn{4}{c}{Graph Similarity} & "
       r"\multicolumn{4}{c}{Output Similarity} & \multicolumn{3}{c}{Consequence $\Delta$} & \multirow{2}{*}{Ask} \\",
       r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}\cmidrule(lr){11-13}",
       r"& & Base & Tone & Lang. & Form. & Base & Tone & Lang. & Form. & Tone & Lang. & Form. & \\", r"\midrule"]
for i, m in enumerate(MODELS):
    for j, (cond, label) in enumerate([("single", "Single-turn"), ("clarify", "Clarification")]):
        g = G[(cond, m)]
        cells = [f3(g[k]) for k in ["Baseline", "Tone", "Language", "Formulation"]]
        cells += [f3(g[k]) for k in ["out_Baseline", "out_Tone", "out_Language", "out_Formulation"]]
        cells += [cons(C[(cond, m, fam)]) for fam in FAMS]
        cells += [f"{float(g['ask_rate']):.2f}" if g["ask_rate"] not in ("", "None") else "--"]
        lead = rf"\multirow{{2}}{{*}}{{{m}}}" if j == 0 else ""
        out.append(f"{lead} & {label} & " + " & ".join(cells) + r" \\")
    if i < len(MODELS) - 1:
        out.append(r"\midrule")
out += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
(D / "table_rq3_clarify.tex").write_text("\n".join(out) + "\n")
print("\n".join(out))
