"""Fixed published tool responses, NOT recorded agent trajectory playback.

Arguments are matched exactly after JSON key ordering only. No coercion,
invented defaults, real HTTP calls, or task-specific oracle filtering.
"""
from __future__ import annotations

import ast
from collections import defaultdict
import copy
import json
from pathlib import Path
import re

from .benchmark_environments import EventLog, fixture_digest, parameters


def route_key(t):
    return (t.get('domain name',''),t.get('parent tool name',''),t.get('API name',''))


def call_key(tool_id, arguments):
    # bool/string/number distinctions preserved; equivalent keys reorder only.
    return fixture_digest({'tool_id':tool_id,'arguments':arguments})


def search_tokens(value):
    """Tokenize natural query text and serialized arguments consistently.

    This lets common coordinate forms such as ``47.8,-121.8`` match separate
    JSON latitude/longitude fields without exposing any task-specific oracle.
    """
    return re.findall(r'-?\d+(?:\.\d+)?|[^\W_]+', str(value).casefold())


def usable_response(t):
    if 'executed_output' not in t or t['executed_output'] is None:
        return False
    if t.get('execution_status') and str(t['execution_status']).lower() not in {'success','successful','ok'}:
        return False
    value=t['executed_output']
    if isinstance(value,str):
        if not value.strip() or value.lstrip().lower().startswith(('error:', 'error ', 'failed:', 'execution failed')):
            return False
        parsed=value
        try:
            parsed=json.loads(value)
        except (ValueError,TypeError):
            try:
                parsed=ast.literal_eval(value)
            except (ValueError,SyntaxError,MemoryError,RecursionError):
                pass
        value=parsed
    if isinstance(value,dict) and (value.get('error') or value.get('errors')):
        return False
    return True


def prepare_schemas(raw_tools):
    groups=defaultdict(list)
    for tool in raw_tools:
        groups[route_key(tool)].append(tool)
    schemas,ambiguous=[],[]
    for route,tools in sorted(groups.items()):
        shapes={json.dumps({k:sorted([{a:b for a,b in p.items() if a not in {'description','default'}}
                             for p in t.get(k,[])],key=lambda p:p['name'])
                            for k in ['required_parameters','optional_parameters']},sort_keys=True)
                for t in tools}
        if len(shapes)>1:
            ambiguous.append({'route':list(route),'reason':'conflicting_parameter_schemas'})
            continue
        selected=min(tools,key=lambda t:json.dumps(t,sort_keys=True))
        record={k:copy.deepcopy(selected[k]) for k in ['tool name','tool description','domain name',
                  'parent tool name','API name','required_parameters','optional_parameters'] if k in selected}
        record['tool_id']='t_'+fixture_digest(list(route))[:20]
        record['aliases']=sorted({t['tool name'] for t in tools})
        record['schema_notes']=[]
        for group in ['required_parameters','optional_parameters']:
            for param in record.get(group,[]):
                versions=[p for t in tools for p in t.get(group,[]) if p['name']==param['name']]
                defaults={json.dumps(p.get('default'),sort_keys=True) for p in versions}
                if len(defaults)>1:
                    param.pop('default',None)
                    record['schema_notes'].append(f"Conflicting published default/example values omitted for {param['name']}; provide explicit arguments")
        schemas.append(record)
    return schemas,ambiguous


class SchemaResolver:
    def __init__(self, schemas):
        self.schemas=schemas
        self.by_name=defaultdict(list)
        for t in schemas:
            for name in t['aliases']:
                self.by_name[name].append(t)

    def resolve(self, tool, domain):
        options=self.by_name.get(tool.get('tool name'),[])
        options=[t for t in options if all(not tool.get(k) or t.get(k)==tool[k]
                  for k in ['domain name','parent tool name','API name'])]
        in_domain=[t for t in options if t.get('domain name')==domain]
        candidates=in_domain or options
        return candidates[0] if len(candidates)==1 else None


def build_library(public_root:Path, output:Path):
    import shutil
    import subprocess
    output.mkdir(parents=True,exist_ok=False)
    raw_tools=json.loads((public_root/'tools/all_tools.json').read_text())
    schemas,ambiguous=prepare_schemas(raw_tools)
    resolver=SchemaResolver(schemas)
    source_files=[]
    index={}
    counts={'source_rows':0,'observed_calls':0,'accepted_observations':0,
            'excluded_missing_or_error_response':0,'unresolved_tool_schema':0}
    files=sorted(list(public_root.glob('parallel/*/*.json'))+list(public_root.glob('sequential/*/*.json')))
    for file in files:
        relative=file.relative_to(public_root)
        data=json.loads(file.read_text())
        if not isinstance(data,list):
            continue
        target=output/'source_snapshot'/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(file,target)
        source_files.append({'path':str(relative),'sha256':fixture_digest(data)})
        for row_index,row in enumerate(data):
            if not isinstance(row,dict) or not isinstance(row.get('tool list'),list):
                continue
            counts['source_rows']+=1
            for tool_index,tool in enumerate(row['tool list']):
                counts['observed_calls']+=1
                if not usable_response(tool):
                    counts['excluded_missing_or_error_response']+=1
                    continue
                schema=resolver.resolve(tool,file.parent.name)
                if schema is None:
                    counts['unresolved_tool_schema']+=1
                    continue
                args=parameters(tool)
                key=call_key(schema['tool_id'],args)
                response=tool['executed_output']
                rhash=fixture_digest(response)
                record=index.setdefault(key,{'tool_id':schema['tool_id'],'arguments':args,'responses':{}})
                item=record['responses'].setdefault(rhash,{'response':response,'count':0,'provenance':[]})
                item['count']+=1
                item['provenance'].append({'file':str(relative),'row':row_index,'tool_index':tool_index})
                counts['accepted_observations']+=1
    fixed={}
    for key,record in index.items():
        # Independent of the selected 40 tasks and any agent outcomes.
        ranked=sorted(record['responses'].items(),key=lambda kv:(-kv[1]['count'],kv[0]))
        rhash,winner=ranked[0]
        fixed[key]={'tool_id':record['tool_id'],'arguments':record['arguments'],
            'response':winner['response'],'response_sha256':rhash,'provenance':winner['provenance'],
            'conflicting_responses':len(ranked)>1,
            'alternatives':[{'sha256':h,'observations':x['count'],'provenance':x['provenance']} for h,x in ranked]}
    package={'schema_version':'traject_fixed_responses_v1','tools':schemas,'responses':fixed}
    (output/'library.json').write_text(json.dumps(package,ensure_ascii=False,separators=(',',':')))
    manifest={**counts,'source_repository':'https://github.com/PengfeiHePower/TRAJECT-Bench',
        'tools':len(schemas),'ambiguous_schemas':ambiguous,'unique_calls':len(fixed),
        'conflicting_call_keys':sum(x['conflicting_responses'] for x in fixed.values()),
        'selection_rule':'Most frequent usable response per exact call; tie broken by response SHA256',
        'cross_call_temporal_coherence':'not_guaranteed; published records may come from different dates',
        'library_sha256':fixture_digest(package),'source_files':source_files,
        'normalization':'JSON key order only; no type coercion, inferred defaults, or argument invention',
        'task_selection_used_for_build':False}
    commit=subprocess.run(['git','-C',str(public_root.parent),'rev-parse','HEAD'],capture_output=True,text=True)
    manifest['source_commit']=commit.stdout.strip() if commit.returncode==0 else None
    license_path=public_root.parent/'LICENSE'
    if license_path.exists():
        shutil.copy2(license_path,output/'SOURCE_LICENSE')
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    return manifest


class FixedResponseEnvironment(EventLog):
    def __init__(self,package,domain):
        super().__init__()
        self.package=package
        self.domain=domain
        self.tools={t['tool_id']:t for t in package['tools'] if t['domain name']==domain}
        self.responses=package['responses']
        # Stable handles let the agent choose an actually observed public call
        # without guessing an argument serialization absent from the library.
        # This catalog spans the complete domain and is not task-specific.
        self.recorded_calls={}
        for key,record in self.responses.items():
            tool_id=record.get('tool_id')
            if tool_id is None:
                # Backward compatibility for early fixtures/libraries whose
                # response row omitted the redundant tool_id field.
                candidates=[candidate for candidate in self.tools
                    if call_key(candidate,record['arguments'])==key]
                tool_id=candidates[0] if len(candidates)==1 else None
            if tool_id in self.tools:
                self.recorded_calls[f'c_{key[:20]}']={'key':key,**record,'tool_id':tool_id}
        self.gaps=[]

    def list_tools(self,query='',offset=0,limit=15):
        if type(offset) is not int or offset<0 or type(limit) is not int or not 1<=limit<=30:
            raise ValueError('Invalid pagination')
        tokens=search_tokens(query)
        matches=[t for t in self.tools.values() if all(
            x in search_tokens(t['tool name']+' '+t.get('tool description','')) for x in tokens)]
        matches.sort(key=lambda t:(t['tool name'],t['tool_id']))
        return {'total':len(matches),'tools':[{'tool_id':t['tool_id'],'name':t['tool name'],
            'description':t.get('tool description','')[:350]} for t in matches[offset:offset+limit]]}

    def describe_tool(self,tool_id):
        t=self.tools.get(tool_id)
        if t is None:
            raise ValueError('Unknown or out-of-domain tool ID')
        return copy.deepcopy({k:v for k,v in t.items() if k!='aliases'})

    def list_recorded_calls(self,tool_id,query='',offset=0,limit=15):
        """List argument combinations backed by fixed published responses."""
        if tool_id not in self.tools:
            raise ValueError('Unknown or out-of-domain tool ID')
        if type(offset) is not int or offset<0 or type(limit) is not int or not 1<=limit<=30:
            raise ValueError('Invalid pagination')
        tokens=search_tokens(query)
        matches=[]
        for call_id,record in self.recorded_calls.items():
            if record['tool_id']!=tool_id:
                continue
            searchable=search_tokens(json.dumps(
                record['arguments'], ensure_ascii=False, sort_keys=True
            ))
            if all(token in searchable for token in tokens):
                matches.append({'call_id':call_id,'arguments':copy.deepcopy(record['arguments'])})
        matches.sort(key=lambda x:(json.dumps(x['arguments'],ensure_ascii=False,sort_keys=True),x['call_id']))
        return {'total':len(matches),'calls':matches[offset:offset+limit],
                'responses_withheld_until_invocation':True}

    def search_recorded_calls(self,query='',argument_query='',offset=0,limit=15):
        """Search call handles across the full domain without task filtering.

        ``query`` matches published tool names/descriptions and
        ``argument_query`` matches the recorded argument object.  Results do
        not expose tool outputs until the agent explicitly invokes a call.
        """
        if type(offset) is not int or offset<0 or type(limit) is not int or not 1<=limit<=30:
            raise ValueError('Invalid pagination')
        tool_tokens=search_tokens(query)
        argument_tokens=search_tokens(argument_query)
        matches=[]
        for call_id,record in self.recorded_calls.items():
            tool=self.tools[record['tool_id']]
            searchable_tool=search_tokens(
                tool['tool name']+' '+tool.get('tool description',''))
            searchable_arguments=search_tokens(json.dumps(
                record['arguments'],ensure_ascii=False,sort_keys=True))
            argument_match=(not record['arguments'] or
                all(token in searchable_arguments for token in argument_tokens))
            if (all(token in searchable_tool for token in tool_tokens) and argument_match):
                matches.append({'call_id':call_id,'tool_id':record['tool_id'],
                    'name':tool['tool name'],'arguments':copy.deepcopy(record['arguments'])})
        matches.sort(key=lambda x:(x['name'],json.dumps(
            x['arguments'],ensure_ascii=False,sort_keys=True),x['call_id']))
        return {'total':len(matches),'calls':matches[offset:offset+limit],
                'responses_withheld_until_invocation':True}

    def invoke_recorded_call(self,call_id):
        record=self.recorded_calls.get(call_id)
        if record is None:
            self.record('invalid_recorded_call_selection',call_id=call_id)
            return {'status':'invalid_recorded_call_selection'}
        tool_id=record['tool_id']
        arguments=copy.deepcopy(record['arguments'])
        self.record('tool_invocation',tool_id=tool_id,tool_name=self.tools[tool_id]['tool name'],
                    arguments=arguments,recorded_call_id=call_id)
        self.record('fixed_tool_response',tool_id=tool_id,response=record['response'],
                    response_sha256=record['response_sha256'],provenance=record['provenance'],
                    conflicting_source_responses=record['conflicting_responses'])
        return {'status':'ok','tool_id':tool_id,'arguments':arguments,
                'result':copy.deepcopy(record['response'])}

    def invoke_tool(self,tool_id,arguments_json):
        if tool_id not in self.tools:
            self.record('invalid_tool_selection',tool_id=tool_id)
            return {'status':'invalid_tool_selection'}
        arguments=json.loads(arguments_json)
        if not isinstance(arguments,dict):
            raise ValueError('arguments_json must encode an object')
        self.record('tool_invocation',tool_id=tool_id,tool_name=self.tools[tool_id]['tool name'],arguments=arguments)
        record=self.responses.get(call_key(tool_id,arguments))
        if record is None:
            schema=self.tools[tool_id]
            required={p['name'] for p in schema.get('required_parameters',[])}
            permitted=required | {p['name'] for p in schema.get('optional_parameters',[])}
            missing=sorted(required-arguments.keys())
            unknown=sorted(arguments.keys()-permitted)
            if missing or unknown:
                self.record('schema_validation_error',tool_id=tool_id,missing=missing,unknown=unknown)
                return {'status':'schema_validation_error','missing_required':missing,'unknown_parameters':unknown,
                        'note':'Validation against published schema; not a response-library coverage judgment'}
            gap={'tool_id':tool_id,'arguments':arguments,'reason':'no_exact_fixed_response'}
            self.gaps.append(gap)
            self.record('coverage_gap',**gap)
            return {'status':'environment_coverage_gap',
                'message':'The fixed experiment response library has no exact entry for these arguments. This is not an agent-error judgment.'}
        self.record('fixed_tool_response',tool_id=tool_id,response=record['response'],
                    response_sha256=record['response_sha256'],provenance=record['provenance'],
                    conflicting_source_responses=record['conflicting_responses'])
        return {'status':'ok','result':copy.deepcopy(record['response'])}

    def dispatch(self,name,arguments):
        self.record('agent_tool_call',tool=name,arguments=arguments)
        if name not in {'list_tools','describe_tool','list_recorded_calls','search_recorded_calls',
                        'invoke_recorded_call','invoke_tool'}:
            result={'status':'invalid_tool_selection'}
        else:
            try:
                result=getattr(self,name)(**arguments)
            except (ValueError,TypeError,KeyError) as exc:
                result={'status':'invalid_tool_request','error_type':type(exc).__name__}
                self.record('invalid_tool_request',tool=name,error_type=type(exc).__name__)
        self.record('agent_tool_result',tool=name,result=result)
        return result


def declarations():
    return [
        {'name':'list_tools','description':'Search the full published tool schema pool for this domain. Does not reveal which inputs have cached responses.',
         'parameters':{'type':'OBJECT','properties':{'query':{'type':'STRING'},'offset':{'type':'INTEGER'},'limit':{'type':'INTEGER'}}}},
        {'name':'describe_tool','description':'Read a tool description and parameter schema. Defaults are published hints, not guaranteed recorded inputs.',
         'parameters':{'type':'OBJECT','properties':{'tool_id':{'type':'STRING'}},'required':['tool_id']}},
        {'name':'list_recorded_calls','description':'List public recorded argument combinations available for one tool. Search with task values, then choose a call yourself. Results are withheld until invocation.',
         'parameters':{'type':'OBJECT','properties':{'tool_id':{'type':'STRING'},'query':{'type':'STRING'},
             'offset':{'type':'INTEGER'},'limit':{'type':'INTEGER'}},'required':['tool_id']}},
        {'name':'search_recorded_calls','description':'Search recorded calls across every tool in the domain. Use query for a specific requested capability and argument_query for task values such as coordinates, IDs, dates, or names. Omit argument_query for parameterless/global calls. Outputs remain withheld until invocation.',
         'parameters':{'type':'OBJECT','properties':{'query':{'type':'STRING'},
             'argument_query':{'type':'STRING'},'offset':{'type':'INTEGER'},
             'limit':{'type':'INTEGER'}}}},
        {'name':'invoke_recorded_call','description':'Invoke one chosen public recorded call by call_id. The agent still chooses the tool, call, and ordering; only published fixed responses are returned.',
         'parameters':{'type':'OBJECT','properties':{'call_id':{'type':'STRING'}},'required':['call_id']}}
    ]
