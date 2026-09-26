#!/usr/bin/env python3
"""Table for RQ2 (consequence consistency with the source prompt), same layout as the earlier compact table."""
import sys as _sys, pathlib as _pl  # scripts are grouped by stage; make sibling stages importable
_sys.path[:0] = [str(_d) for _d in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _d.is_dir()]
import csv
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "results/rq2_consequence"
rows = {(r["dataset"], r["model"], r["family"]): r for r in csv.DictReader((D / "consistency_summary.csv").open())}
CONS = [("AI Agent Permissions", "Permission decision"), ("AgentCIBench", "Shared information"),
        ("TRAJECT-Bench", "Tool invocations")]
MODELS = [("Gemini 2.5 Flash", "Gemini"), ("GPT-4.1 Mini", "GPT-4.1m"), ("Claude Haiku 4.5", "Haiku")]
FAMS = ["Tone", "Language", "Formulation"]
f = lambda x: f"{float(x):+.2f}"
out = [r"\begin{table}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
       r"\caption{Consequence inconsistency with the source prompt. S--S: fraction of within-source execution pairs whose "
       r"consequence differs. $\Delta$: source--variant minus within-source inconsistency, mean $\pm$ s.e.\ over tasks; "
       r"bold marks a 95\% task-level bootstrap CI excluding 0. Consequences: requested (field, scope) set (AIP); "
       r"output channels and disclosed information items (ACB); multiset of invoked tools (TJB).}",
       r"\label{tab:rq2_consistency}", r"\begin{tabular}{llcccc}", r"\toprule",
       r"Consequence & Model & S--S & Tone & Language & Formulation \\", r"\midrule"]
for i, (ds, name) in enumerate(CONS):
    for j, (m, short) in enumerate(MODELS):
        cells = []
        for fam in FAMS:
            r = rows[(ds, m, fam)]
            c = rf"{f(r['delta'])}\,$\pm$\,{float(r['se']):.2f}"
            cells.append(rf"\textbf{{{c}}}" if float(r["ci_low"]) > 0 or float(r["ci_high"]) < 0 else c)
        lead = rf"\multirow{{3}}{{*}}{{{name}}}" if j == 0 else ""
        out.append(f"{lead} & {short} & {float(rows[(ds, m, 'Tone')]['within_source']):.2f} & " + " & ".join(cells) + r" \\")
    if i < 2:
        out.append(r"\midrule")
out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
(D / "table_rq2_consistency.tex").write_text("\n".join(out) + "\n")
print("\n".join(out))
