#!/usr/bin/env bash
# Recompute every table and figure of the paper from the shipped runs and LLM-judge scores (no API calls).
set -euo pipefail
cd "$(dirname "$0")"
S=scripts

# 0. restore the agent run directories from runs/**/*.jsonl.gz (first time only)
[ -d runs/main/claude-haiku-4.5 ] || python3 $S/unpack_runs.py

# RQ1: operational (graph) consistency  -> Table 3, Figure 2, Figure 3
python3 $S/04_analysis/compute_rq1_fullgraph_3x3.py
python3 $S/05_tables_figures/render_table3.py
python3 $S/05_tables_figures/plot_rq1_similarity_distributions.py --subset all
python3 $S/05_tables_figures/plot_example_graphs.py

# RQ1: output consistency (GPT-4o scores in results/rq1_output/evaluator_results.jsonl) -> output table, scatter
python3 $S/04_analysis/analyze_output_similarity.py
python3 $S/05_tables_figures/plot_output_vs_graph.py

# RQ2: consequence consistency with the source prompt -> RQ2 table and figure
python3 $S/04_analysis/analyze_rq2_consequences.py
rm -rf "$HOME/.rq2_consistency_cache"
python3 $S/04_analysis/analyze_rq2_consistency.py
python3 $S/05_tables_figures/render_rq2_consistency_latex.py
python3 $S/05_tables_figures/plot_rq2_consistency.py

# RQ3: clarification and GPT-6 Astra on the AgentCIBench subset
python3 $S/04_analysis/analyze_rq3_acb.py
python3 $S/05_tables_figures/render_rq3_clarify_table.py
python3 $S/05_tables_figures/plot_rq3_gpt6.py

# RQ3: coding agents on Terminal-Bench 2.0
python3 $S/04_analysis/analyze_tb2.py
python3 $S/04_analysis/analyze_tb2_output.py
python3 $S/05_tables_figures/render_tb2_table.py

# Mitigation: GPT-4o information extraction
python3 $S/04_analysis/analyze_mitigation.py
python3 $S/05_tables_figures/render_mitigation_table.py

echo "done: tables (*.tex) and figures (*.pdf) are under results/"
