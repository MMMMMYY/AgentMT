"""One explicit retry for incomplete preparation; preserve the failed attempt."""
import sys as _sys, pathlib as _pl  # scripts are grouped by stage; make sibling stages importable
_sys.path[:0] = [str(_d) for _d in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _d.is_dir()]
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from privacy_mt.prepare_multilingual_variants import prepare,compile_all
from privacy_mt.benchmark_environments import fixture_digest


def main():
    out=ROOT/'dataset/variants/multilingual'
    rows=json.loads((ROOT/'dataset/environments/fixed_sources.json').read_text())['unified']
    pending=[]
    for row in rows:
        folder=out/row['taskId']
        suite=json.loads((folder/'suite.json').read_text())
        if suite['status']=='reviewed': continue
        archive=folder/'attempt1_incomplete'
        archive.mkdir(exist_ok=False)
        # Keep successful translations cached; retry only failed generation/review.
        names=['suite.json']
        names+=['reviewer_raw.json','reviewer_request.json'] if suite['variants'] else ['translator_raw.json','translator_request.json']
        for name in names:
            if (folder/name).exists(): (folder/name).rename(archive/name)
        pending.append(row)
    (out/'retry1_manifest.json').write_text(json.dumps({'task_ids':[r['taskId'] for r in pending],
        'source_sha256':fixture_digest(rows),'script':Path(__file__).read_text()},ensure_ascii=False,indent=2)+'\n')
    with ThreadPoolExecutor(max_workers=3) as pool:
        for f in as_completed([pool.submit(prepare,row,out) for row in pending]):
            s=f.result();print(json.dumps({'task_id':s['task_id'],'status':s['status'],'variants':len(s['variants'])}),flush=True)
    print(json.dumps(compile_all(rows,out)))


if __name__=='__main__': main()
