"""Run the three-dataset operational-consistency experiment.

Each episode receives one user prompt and tool observations only.  Dataset
specific environments emit a graph plus a policy-relevant projection.  The
runner is append-only: existing completed episode directories are reused and
never overwritten.
"""
from __future__ import annotations
import sys as _sys, pathlib as _pl  # scripts are grouped by stage; make sibling stages importable
_sys.path[:0] = [str(_d) for _d in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _d.is_dir()]

import argparse
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import json
from pathlib import Path
from queue import Queue
import shutil
import sys
from threading import Lock
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

from privacy_mt.aip_permission_agent import run_episode as run_aip
from privacy_mt.benchmark_environments import fixture_digest
from privacy_mt.fixed_agent_runner import create_client, run_episode as run_agentci, write_json
from privacy_mt.traject_fixed_runner import run_episode as run_traject, task_id as traject_task_id
from privacy_mt.operation_similarity import build_operation_graph, load_events
from privacy_mt.openai_agent_runner import (
    DEFAULT_MODEL as DEFAULT_OPENAI_MODEL,
    OpenAIResponsesClient,
    run_agentci_episode as run_openai_agentci,
    run_aip_episode as run_openai_aip,
    run_traject_episode as run_openai_traject,
)
from privacy_mt.anthropic_agent_runner import (
    DEFAULT_MODEL as DEFAULT_ANTHROPIC_MODEL,
    AnthropicMessagesClient,
    run_agentci_episode as run_anthropic_agentci,
    run_aip_episode as run_anthropic_aip,
    run_traject_episode as run_anthropic_traject,
)


OPENAI_PRICE_PER_MILLION = {
    'input': 0.40,
    'cached_input': 0.10,
    'output': 1.60,
}


def estimated_openai_cost(result):
    total=0.0
    for usage in result.get('usage',[]):
        input_tokens=int(usage.get('input_tokens',0) or 0)
        cached=int(usage.get('input_tokens_details',{}).get('cached_tokens',0) or 0)
        output_tokens=int(usage.get('output_tokens',0) or 0)
        total += ((input_tokens-cached)*OPENAI_PRICE_PER_MILLION['input'] +
                  cached*OPENAI_PRICE_PER_MILLION['cached_input'] +
                  output_tokens*OPENAI_PRICE_PER_MILLION['output'])/1_000_000
    return total


def load_formulation_variants():
    """Return exactly one formulation candidate per source task.

    Lossless sentence reordering is preferred.  The remaining atomic prompts
    use the first locked-span GPT-4o paraphrase.  Surface-check failures remain
    explicitly marked so they can be excluded from strict MT analysis without
    losing experimental coverage.
    """
    selected = {}
    reorder = json.loads((ROOT/'dataset/variants/formulation_reordering/manifest.json').read_text())
    for row in reorder:
        if row['status'] == 'ready':
            selected[row['task_id']] = {
                'id': 'formulation', 'prompt': row['variant']['prompt'],
                'subtype': 'information_reordering', 'method': 'NL-Augmenter-style sentence permutation',
                'strict_mt_eligible': True,
            }
    paraphrases = json.loads((ROOT/'dataset/variants/formulation_paraphrase/candidates.json').read_text())
    for row in paraphrases:
        if row['task_id'] in selected:
            continue
        candidate = next((value for value in row['candidates']
                          if value['deterministic_checks']['passed'] and
                          not value.get('missing_locked_spans')), row['candidates'][0])
        selected[row['task_id']] = {
            'id': 'formulation', 'prompt': candidate['prompt'],
            'subtype': 'constrained_paraphrasing',
            'method': 'GPT-4o locked-span paraphrasing',
            'strict_mt_eligible': bool(candidate['deterministic_checks']['passed'] and
                                       not candidate.get('missing_locked_spans')),
            'surface_failures': candidate['deterministic_checks']['failures'],
        }
    if len(selected) != 116:
        raise ValueError(f'Expected formulation coverage for 116 tasks, found {len(selected)}')
    return selected


def load_plan(families, datasets, task_filter=None):
    sources=json.loads((ROOT/'dataset/environments/fixed_sources.json').read_text())
    rows={}
    for row in sources['aiapRows']:
        rows[f'AIP-{row["id"]:03}']={'dataset':'AI Agent Permissions','row':row,'source':row['query']}
    for row in sources['agentCiRows']:
        rows[f'ACB-{row["scenario_id"]}']={'dataset':'AgentCIBench','row':row,'source':row['task_prompt']}
    for row in sources['trajectRows']:
        rows[traject_task_id(row)]={'dataset':'TRAJECT-Bench','row':row,'source':row['query']}
    plan=[]
    formulation = load_formulation_variants() if 'formulation' in families else {}
    for tid,item in sorted(rows.items()):
        if item['dataset'] not in datasets or task_filter and tid not in task_filter:
            continue
        if 'source' in families:
            plan.append({**item,'task_id':tid,'form':'source','family':'source','prompt':item['source'],
                         'strict_mt_eligible':True})
        if 'tone' in families:
            suite=json.loads((ROOT/'dataset/variants/tone'/tid/'suite_ready_v3.json').read_text())
            for variant in suite['variants']:
                plan.append({**item,'task_id':tid,'form':variant['id'],'family':'tone',
                    'prompt':variant['prompt'],'tone':variant.get('tone'),
                    'strict_mt_eligible':bool(variant.get('eligible_pilot') is True and
                                              not variant.get('context_sensitive',False))})
        if 'multilingual' in families:
            suite=json.loads((ROOT/'dataset/variants/multilingual'/tid/'suite.json').read_text())
            for variant in suite['variants']:
                plan.append({**item,'task_id':tid,'form':variant['id'],'family':'multilingual',
                    'prompt':variant['prompt'],'language':variant.get('language'),
                    'strict_mt_eligible':bool(variant.get('eligible_pilot') is True)})
        if 'information_order' in families:
            family_map={x['task_id']:x for x in json.loads(
                (ROOT/'dataset/variants/information_order/families.json').read_text())}
            for variant in family_map[tid]['variants']:
                if variant.get('eligible_pilot') is True:
                    plan.append({**item,'task_id':tid,'form':variant['id'],
                        'family':'information_order','prompt':variant['prompt'],
                        'strict_mt_eligible':bool(variant.get('eligible_pilot') is True)})
        if 'formulation' in families:
            variant = formulation[tid]
            plan.append({**item,'task_id':tid,'form':variant['id'],'family':'formulation',
                         'prompt':variant['prompt'], **{k:v for k,v in variant.items() if k not in {'id','prompt'}}})
    return plan


def infer_task_id(path, payload):
    if payload.get('task_id'):
        return payload['task_id']
    if payload.get('scenario_id'):
        return 'ACB-'+payload['scenario_id']
    for part in reversed(path.parts):
        if part.startswith(('AIP-','ACB-','TJB-')):
            return part
    return None


def reusable_episodes(output):
    """Index prior completed Gemini episodes by exact task and prompt."""
    index = {}
    artifacts = ROOT/'runs'
    for input_path in artifacts.glob('**/input.json'):
        result_path=input_path.parent/'result.json'
        if (output in input_path.parents or not result_path.exists() or
                not (input_path.parent/'events.jsonl').exists() or
                not (input_path.parent/'raw_responses.jsonl').exists()):
            continue
        try:
            payload = json.loads(input_path.read_text())
            result = json.loads(result_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if result.get('status') == 'infrastructure_error':
            continue
        model = payload.get('model') or payload.get('protocol',{}).get('model')
        task = infer_task_id(input_path, payload)
        prompt = payload.get('prompt')
        if task and prompt and model == 'gemini-2.5-flash':
            index.setdefault((task,prompt),[]).append(input_path.parent)
    for paths in index.values():
        paths.sort(key=str)
    return index


def materialize_reuse(source, destination):
    shutil.copytree(source,destination)
    write_json(destination/'reuse.json',{'reused_from':str(source.relative_to(ROOT)),
        'basis':'exact task_id, prompt, model, and completed event trace'})


def write_operation_graph(directory):
    events = load_events(directory/'events.jsonl')
    graph = build_operation_graph(events,name=directory.name)
    write_json(directory/'operation_graph.json',graph.to_dict())


def completed_episode(directory):
    result_path=directory/'result.json'
    if (not result_path.exists() or not (directory/'input.json').exists() or
            not (directory/'raw_responses.jsonl').exists() or
            not (directory/'events.jsonl').exists()):
        return False
    try:
        return json.loads(result_path.read_text()).get('status') != 'infrastructure_error'
    except (OSError,json.JSONDecodeError):
        return False


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--families',nargs='+',choices=['source','tone','multilingual','information_order','formulation'],
                        default=['source'])
    parser.add_argument('--datasets',nargs='+',choices=['AI Agent Permissions','AgentCIBench','TRAJECT-Bench'],
                        default=['AI Agent Permissions','AgentCIBench','TRAJECT-Bench'])
    parser.add_argument('--tasks',nargs='*')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--pending-limit',type=int,
        help='Run at most this many currently incomplete episodes after resume scanning')
    parser.add_argument('--repeats',type=int,default=1)
    parser.add_argument('--reuse-existing',action='store_true',
        help='DEPRECATED: reuse is disabled because it ignored protocol_sha256 (see scripts/fix_gemini_protocol_mismatch.py)')
    parser.add_argument('--provider',choices=['gemini','openai','anthropic'],default='gemini')
    parser.add_argument('--model')
    parser.add_argument('--api-key-file',type=Path)
    parser.add_argument('--max-estimated-cost',type=float)
    parser.add_argument('--project',default='YOUR_GCP_PROJECT')
    parser.add_argument('--location',default='us-central1')
    parser.add_argument('--forms',nargs='*',help='restrict to these prompt forms (e.g. source p01 lang_zh formulation)')
    parser.add_argument('--prompt-override',type=Path,
        help='jsonl with task_id/form/extracted: run the extracted prompt instead of the original (mitigation)')
    parser.add_argument('--library',type=Path,
        default=Path('dataset/traject_response_library'))
    args=parser.parse_args()
    if args.model is None:
        args.model = {'gemini':'gemini-2.5-flash','openai':DEFAULT_OPENAI_MODEL,
                      'anthropic':DEFAULT_ANTHROPIC_MODEL}[args.provider]
    if args.max_estimated_cost is not None and args.provider!='openai':
        raise SystemExit('--max-estimated-cost is currently defined only for OpenAI runs')
    args.output=(args.output if args.output.is_absolute() else ROOT/args.output).resolve()
    if not 1<=args.workers<=24:
        raise SystemExit('workers must be between 1 and 24')
    if not 1<=args.repeats<=20:
        raise SystemExit('repeats must be between 1 and 20')
    if args.pending_limit is not None and args.pending_limit < 1:
        raise SystemExit('pending-limit must be positive')
    plan=load_plan(set(args.families),set(args.datasets),set(args.tasks) if args.tasks else None)
    if args.forms:
        plan=[x for x in plan if x['form'] in set(args.forms)]
    if args.prompt_override:
        override={}
        for line in args.prompt_override.read_text(encoding='utf-8').splitlines():
            r=json.loads(line); override[(r['task_id'],r['form'])]=r
        for x in plan:
            r=override[(x['task_id'],x['form'])]
            if r['prompt']!=x['prompt']:
                raise SystemExit(f"override was extracted from a different prompt: {x['task_id']} {x['form']}")
            x['original_prompt']=x['prompt']; x['prompt']=r['extracted']
            x['prompt_override_sha256']=r['system_sha256']
    if args.limit is not None:
        plan=plan[:args.limit]
    library=json.loads((ROOT/args.library/'library.json').read_text())
    library_hash=fixture_digest(library)
    expected=json.loads((ROOT/args.library/'manifest.json').read_text())['library_sha256']
    if library_hash!=expected:
        raise SystemExit('TRAJECT library checksum mismatch')
    args.output.mkdir(parents=True,exist_ok=True)
    manifest_rows=[{k:x[k] for k in x if k not in {'row','source','dataset'}} for x in plan]
    manifest={'schema':'full_graph_experiment_v3','provider':args.provider,'model':args.model,
        'families':args.families,
        'datasets':args.datasets,'repeats':args.repeats,'episodes':manifest_rows,
        'plan_sha256':fixture_digest(manifest_rows),
        'traject_library_sha256':library_hash,
        'max_estimated_cost_usd':args.max_estimated_cost,
        'openai_price_per_million_tokens':(
            OPENAI_PRICE_PER_MILLION if args.provider=='openai' else None)}
    manifest_path=args.output/'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text())!=manifest:
        raise SystemExit('Existing output manifest differs; choose a new output directory')
    if not manifest_path.exists():
        write_json(manifest_path,manifest)
    pending=[]
    reused=0
    if args.reuse_existing:
        raise SystemExit('--reuse-existing is disabled: it copied runs across protocol versions')
    reuse_index={}
    used_sources=set()
    for reuse_file in args.output.glob('*/*/r*/reuse.json'):
        try:
            origin=Path(json.loads(reuse_file.read_text())['reused_from'])
            used_sources.add((origin if origin.is_absolute() else ROOT/origin).resolve())
        except (OSError,KeyError,json.JSONDecodeError):
            pass
    for item in plan:
        for repeat in range(args.repeats):
            directory=args.output/item['task_id']/f'{item["family"]}__{item["form"]}'/f'r{repeat}'
            if completed_episode(directory):
                if not (directory/'operation_graph.json').exists():
                    write_operation_graph(directory)
                continue
            if directory.exists():
                failed=args.output/'failed_attempts'/item['task_id']/f'{item["family"]}__{item["form"]}'/f'r{repeat}_{int(time.time())}'
                failed.parent.mkdir(parents=True,exist_ok=True)
                shutil.move(directory,failed)
            candidates=[p.resolve() for p in reuse_index.get((item['task_id'],item['prompt']),[])
                        if p.resolve() not in used_sources]
            if candidates:
                source=candidates[0]
                used_sources.add(source)
                directory.parent.mkdir(parents=True,exist_ok=True)
                materialize_reuse(source,directory)
                write_operation_graph(directory)
                reused+=1
                continue
            pending.append((item,directory,repeat))
    total=len(plan)*args.repeats
    remaining_pending=len(pending)
    if args.pending_limit is not None:
        pending=pending[:args.pending_limit]
    complete=total-remaining_pending
    print(json.dumps({'planned_forms':len(plan),'planned_runs':total,'already_complete_or_reused':complete,
                      'reused_this_start':reused,'remaining_incomplete':remaining_pending,
                      'scheduled_this_start':len(pending)}),flush=True)
    clients=Queue()
    lock=Lock()
    estimated_cost_usd=0.0
    summary_path=args.output/'run_summary.jsonl'
    with ExitStack() as stack:
        def new_client():
            if args.provider=='gemini':
                return create_client(args.project,args.location)
            if args.provider=='openai':
                return OpenAIResponsesClient.from_environment(args.api_key_file)
            return AnthropicMessagesClient.from_environment(args.api_key_file)
        for _ in range(min(args.workers,max(1,len(pending)))):
            clients.put((stack.enter_context(new_client()),time.monotonic()))
        def execute(entry):
            nonlocal estimated_cost_usd
            item,directory,repeat=entry
            with lock:
                if (args.max_estimated_cost is not None and
                        estimated_cost_usd>=args.max_estimated_cost):
                    record={'task_id':item['task_id'],'dataset':item['dataset'],
                        'family':item['family'],'form':item['form'],'repeat':repeat,
                        'status':'estimated_cost_budget'}
                    with summary_path.open('a') as out:
                        out.write(json.dumps(record,ensure_ascii=False)+'\n')
                    return record
            client,born=clients.get()
            # create_client uses a bounded gcloud access token.  Replace it
            # before the one-hour token lifetime during long campaigns.
            if args.provider=='gemini' and time.monotonic()-born > 2400:
                try:
                    client.close()
                except Exception:
                    pass
                with lock:
                    client=stack.enter_context(new_client())
                born=time.monotonic()
            try:
                if item['dataset']=='AI Agent Permissions':
                    if args.provider=='gemini':
                        result=run_aip(client,item['row'],item['prompt'],directory,args.model)
                    elif args.provider=='openai':
                        result=run_openai_aip(client,item['row'],item['prompt'],directory,model=args.model)
                    else:
                        result=run_anthropic_aip(client,item['row'],item['prompt'],directory,model=args.model)
                elif item['dataset']=='AgentCIBench':
                    if args.provider=='gemini':
                        result=run_agentci(client,item['row'],item['prompt'],directory,model=args.model)
                    elif args.provider=='openai':
                        result=run_openai_agentci(client,item['row'],item['prompt'],directory,model=args.model)
                    else:
                        result=run_anthropic_agentci(client,item['row'],item['prompt'],directory,model=args.model)
                else:
                    if args.provider=='gemini':
                        result=run_traject(client,item['row'],item['prompt'],library,library_hash,directory)
                    elif args.provider=='openai':
                        result=run_openai_traject(client,item['row'],item['prompt'],library,library_hash,
                                                  directory,model=args.model)
                    else:
                        result=run_anthropic_traject(client,item['row'],item['prompt'],library,library_hash,
                                                     directory,model=args.model)
                write_operation_graph(directory)
                episode_cost=(estimated_openai_cost(result) if args.provider=='openai' else 0.0)
                with lock:
                    estimated_cost_usd+=episode_cost
                record={'task_id':item['task_id'],'dataset':item['dataset'],'family':item['family'],
                    'form':item['form'],'repeat':repeat,'status':result['status'],
                    'estimated_cost_usd':episode_cost,
                    'cumulative_estimated_cost_usd':estimated_cost_usd}
            except Exception as exc:
                record={'task_id':item['task_id'],'dataset':item['dataset'],'family':item['family'],
                    'form':item['form'],'repeat':repeat,'status':'runner_exception','error_type':type(exc).__name__}
            finally:
                clients.put((client,born))
            with lock:
                with summary_path.open('a') as out:
                    out.write(json.dumps(record,ensure_ascii=False)+'\n')
            return record
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(execute,item) for item in pending]
            for future in as_completed(futures):
                print(json.dumps(future.result(),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
