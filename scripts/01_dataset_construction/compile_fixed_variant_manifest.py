"""Flatten all candidates and apply deterministic protected-literal checks.

Does not promote automated semantic review to human-verified equivalence.
"""
import sys as _sys, pathlib as _pl  # scripts are grouped by stage; make sibling stages importable
_sys.path[:0] = [str(_d) for _d in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _d.is_dir()]
import copy
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from privacy_mt.fixed_agent_runner import write_json
from privacy_mt.prepare_fixed_variants import numeric_tokens


def protected_strings(text):
    literals=set(re.findall(r'"([^"\n]+)"',text))
    literals.update(re.findall(r'“([^”\n]+)”',text))
    literals.update(re.findall(r"(?<!\w)'([^'\n]+)'",text))
    literals.update(re.findall(r'https?://[^\s<>"\']+',text))
    literals.update(re.findall(r'\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b',text))
    return {x.rstrip('.?,;') for x in literals if x.strip()}


def main():
    root=ROOT/'dataset/variants/tone'
    records=[]
    families=[]
    for folder in sorted(p.parent for p in root.glob('*/suite_reviewed_v2.json')):
        source=json.loads((folder/'source.json').read_text())
        suite=copy.deepcopy(json.loads((folder/'suite_reviewed_v2.json').read_text()))
        literals=protected_strings(source['prompt'])
        for variant in suite['variants']:
            missing=sorted(x for x in literals if x not in variant['prompt'])
            missing_numbers=sorted(numeric_tokens(source['prompt'])-numeric_tokens(variant['prompt']))
            variant['protected_literal_check']={'passed':not missing and not missing_numbers,
                'missing_strings':missing,'missing_numbers':missing_numbers}
            variant['eligible_pilot']=variant['eligible_pilot'] and not missing and not missing_numbers
            records.append({'task_id':source['taskId'],'dataset':source['dataset'],
                'source_prompt':source['prompt'],**variant})
        suite['deterministic_literal_audit']='v1'
        suite['status']='candidates_generated'
        write_json(folder/'suite_ready_v3.json',suite)
        families.append({'task_id':source['taskId'],'dataset':source['dataset'],
            'candidates':len(suite['variants']),'eligible_pilot':sum(v['eligible_pilot'] for v in suite['variants']),
            'literal_check_failures':sum(not v['protected_literal_check']['passed'] for v in suite['variants'])})
    if len(families)!=116 or len(records)!=1740:
        raise ValueError('Incomplete candidate population')
    (root/'all_candidates.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    summary={'source_tasks':len(families),'candidate_variants':len(records),
        'eligible_after_model_and_literal_checks':sum(r['eligible_pilot'] for r in records),
        'context_sensitive_candidates':sum(r['context_sensitive'] for r in records),
        'literal_check_failures':sum(not r['protected_literal_check']['passed'] for r in records),
        'families':families,'independent_human_validation':False,
        'all_environments_ready':False}
    write_json(root/'compiled_summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='families'}))


if __name__=='__main__':
    main()
