"""Readable export and Unicode-aware literal audit; never overwrite generation records."""
import sys as _sys, pathlib as _pl  # scripts are grouped by stage; make sibling stages importable
_sys.path[:0] = [str(_d) for _d in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _d.is_dir()]
import json
from pathlib import Path
import re
import sys
from collections import Counter

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from privacy_mt.prepare_multilingual_variants import CHECKS,LANGUAGES,protected_strings


def numeric_tokens(text):
    # CJK/Cyrillic/Arabic letters adjacent to a digit do not hide that digit.
    return set(re.findall(r'(?<![A-Za-z0-9_])[+-]?\d+(?:[.,:]\d+)*(?:st|nd|rd|th)?',text))


def main():
    out=ROOT/'dataset/variants/multilingual'
    rows=json.loads((ROOT/'dataset/environments/fixed_sources.json').read_text())['unified']
    records=[];queue=[];docs=['# 多语言 source prompt 变体\n',
        '英语原文直接翻译为七种语言，不叠加语气或信息顺序变形。自动检查不等于人工确认；待复核项不可直接用于正式实验。\n']
    for row in rows:
        s=json.loads((out/row['taskId']/'suite.json').read_text())
        assert s['source_prompt']==row['prompt']
        docs.extend([f'## {row["taskId"]} — {row["dataset"]}\n', '### English — source\n',row['prompt']+'\n'])
        records.append({'task_id':row['taskId'],'dataset':row['dataset'],'language':'en',
                        'prompt':row['prompt'],'is_translation':False,'status':'source'})
        for v in s['variants']:
            missing=[x for x in protected_strings(row['prompt']) if x not in v['prompt']]
            numbers=sorted(numeric_tokens(row['prompt'])-numeric_tokens(v['prompt']))
            review=v.get('review',{})
            passed=not missing and not numbers and all(review.get(k) is True for k in CHECKS)
            record={'task_id':row['taskId'],'dataset':row['dataset'],'source_prompt':row['prompt'],**v,
                    'is_translation':True,'eligible_pilot':passed,
                    'literal_check_v2':{'passed':not missing and not numbers,'missing_strings':missing,'missing_numeric_tokens':numbers},
                    'status':'automated_checks_passed' if passed else 'needs_review'}
            records.append(record)
            if not passed: queue.append(record)
            docs.extend([f'### {LANGUAGES[v["language"]]} — {record["status"]}\n',v['prompt']+'\n'])
    (out/'all_prompts_audited_v2.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    (out/'PROMPTS.md').write_text('\n'.join(docs))
    (out/'review_queue_v2.json').write_text(json.dumps(queue,ensure_ascii=False,indent=2)+'\n')
    summary={'source_tasks':len(rows),'translations':sum(r['is_translation'] for r in records),
             'total_with_english':len(records),'needs_review':len(queue),
             'automated_checks_passed':sum(r['status']=='automated_checks_passed' for r in records),
             'counts_by_language':dict(Counter(r['language'] for r in records)),
             'source_unchanged':True,'human_validated':False,'agent_experiments_run':False,
             'audit_note':'v2 corrects Unicode-adjacent numeric detection; generation and v1 audits remain unchanged.'}
    (out/'audit_summary_v2.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary))


if __name__=='__main__': main()
