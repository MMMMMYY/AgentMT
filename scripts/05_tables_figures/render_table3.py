"""Table 3 (all executed variants) from compute_rq1_fullgraph_3x3 output."""
import sys as _sys, pathlib as _pl  # scripts are grouped by stage; make sibling stages importable
_sys.path[:0] = [str(_d) for _d in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _d.is_dir()]
import csv
D = 'results/rq1_graph'
M = ['Gemini 2.5 Flash', 'GPT-4.1 Mini', 'Claude Haiku 4.5']
DS = ['AI Agent Permissions', 'AgentCIBench', 'TRAJECT-Bench']
rows = {(r['dataset'], r['model']): r for r in csv.DictReader(open(f'{D}/table_all.csv')) if r['subset'] == 'all'}
out = [['Dataset', 'Model', 'Baseline', 'Tone', 'Language', 'Formulation', 'Tone Δ', 'Language Δ', 'Formulation Δ']]
tex = []
for d in DS:
    for i, m in enumerate(M):
        r = rows[(d, m)]; f = lambda k: f"{float(r[k]):.3f}"
        vals = [f('baseline_graph_similarity_mean'), f('tone_source_variant_similarity_mean'),
                f('multilingual_source_variant_similarity_mean'), f('formulation_source_variant_similarity_mean')]
        out.append([d, m, *vals, f('tone_delta_mean'), f('multilingual_delta_mean'), f('formulation_delta_mean')])
        tex.append((rf'\multirow{{3}}{{*}}{{{d}}}' if i == 0 else '') + f' & {m:<16} & ' + ' & '.join(vals) + r' \\')
    tex.append(r'\midrule')
csv.writer(open(f'{D}/table_similarity_means_all.csv', 'w', newline='')).writerows(out)
open(f'{D}/table3_rows.tex', 'w').write('\n'.join(tex[:-1]) + '\n')
for o in out: print(' | '.join(o))
