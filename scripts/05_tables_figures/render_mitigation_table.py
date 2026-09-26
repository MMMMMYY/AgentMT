#!/usr/bin/env python3
"""LaTeX table: no mitigation vs GPT-4o information extraction (results/mitigation)."""
import sys as _sys, pathlib as _pl  # scripts are grouped by stage; make sibling stages importable
_sys.path[:0] = [str(_d) for _d in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _d.is_dir()]
import csv
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "results/mitigation"
G = {(r["model"], r["dataset"], r["condition"]): r for r in csv.DictReader((D / "graph.csv").open())}
O = {(r["model"], r["dataset"], r["condition"]): r for r in csv.DictReader((D / "output.csv").open())}
C = {(r["model"], r["dataset"], r["condition"], r["family"]): r for r in csv.DictReader((D / "consequence.csv").open())}
MODELS = [("Gemini 2.5 Flash", "Gemini"), ("GPT-4.1 Mini", "GPT-4.1m"), ("Claude Haiku 4.5", "Haiku")]
DATASETS = [("AI Agent Permissions", "AIP"), ("AgentCIBench", "ACB"), ("TRAJECT-Bench", "TJB")]
FAMS = ["Tone", "Language", "Formulation"]
f3 = lambda x: f"{float(x):.2f}"

def cons(r):
    c = rf"{float(r['delta']):+.2f}\,$\pm$\,{float(r['se']):.2f}"
    return rf"\textbf{{{c}}}" if float(r["ci_low"]) > 0 or float(r["ci_high"]) < 0 else c

out = [r"\begin{table*}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{2.5pt}",
       r"\caption{Consistency without mitigation and with GPT-4o information extraction (10 tasks per benchmark, "
       r"8 variants, 3 runs each). With mitigation, every prompt---source and variants---is rewritten once into a "
       r"normalized English task specification before execution. Graph and output columns follow "
       r"Table~\ref{tab:rq1} (output: source--variant similarity); consequence columns report $\Delta$ as in "
       r"Table~\ref{tab:rq2_consistency} (mean $\pm$ s.e.; bold: 95\% bootstrap CI excludes 0). "
       r"Rows without mitigation reuse the main-study executions.}",
       r"\label{tab:mitigation}", r"\begin{tabular}{lllccccccccccc}", r"\toprule",
       r"\multirow{2}{*}{Model} & \multirow{2}{*}{Bench.} & \multirow{2}{*}{Mitig.} & \multicolumn{4}{c}{Graph Similarity} & "
       r"\multicolumn{4}{c}{Output Similarity} & \multicolumn{3}{c}{Consequence $\Delta$} \\",
       r"\cmidrule(lr){4-7}\cmidrule(lr){8-11}\cmidrule(lr){12-14}",
       r"& & & Base & Tone & Lang. & Form. & Base & Tone & Lang. & Form. & Tone & Lang. & Form. \\", r"\midrule"]
for i, (m, ms) in enumerate(MODELS):
    for j, (ds, dss) in enumerate(DATASETS):
        for k, (cond, label) in enumerate([("none", "--"), ("mitigation", r"\checkmark")]):
            g, o = G[(m, ds, cond)], O[(m, ds, cond)]
            cells = [f3(g[x]) for x in ["Baseline"] + FAMS] + [f3(o[x]) for x in ["Baseline"] + FAMS]
            cells += [cons(C[(m, ds, cond, fam)]) for fam in FAMS]
            lead_m = rf"\multirow{{6}}{{*}}{{{ms}}}" if j == 0 and k == 0 else ""
            lead_d = rf"\multirow{{2}}{{*}}{{{dss}}}" if k == 0 else ""
            out.append(f"{lead_m} & {lead_d} & {label} & " + " & ".join(cells) + r" \\")
        if j < 2:
            out.append(r"\cmidrule(lr){2-14}")
    if i < 2:
        out.append(r"\midrule")
out += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
(D / "table_mitigation.tex").write_text("\n".join(out) + "\n")
print("\n".join(out))
