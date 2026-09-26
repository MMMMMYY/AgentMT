#!/usr/bin/env python3
"""Mitigation step: GPT-4o rewrites every prompt of the mitigation subset into a normalized English task
specification (prompts/mitigation_extraction_system.txt). One extraction per (task, form), temperature 0,
shared by all agent models. Resumable. Output: dataset/mitigation_extraction/extractions.jsonl

  python3 scripts/01_dataset_construction/extract_mitigation_prompts.py --api-key-file secrets/openai_key.txt --sample 3   # 3 tasks per benchmark
  python3 scripts/01_dataset_construction/extract_mitigation_prompts.py --api-key-file secrets/openai_key.txt              # all 270
"""
import sys as _sys, pathlib as _pl  # scripts are grouped by stage; make sibling stages importable
_sys.path[:0] = [str(_d) for _d in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _d.is_dir()]
import argparse, hashlib, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path[:0] = [str(p) for p in sorted((ROOT / "scripts").iterdir()) if p.is_dir()]
from run_full_graph_experiment import load_plan  # noqa: E402
from privacy_mt.openai_agent_runner import OpenAIResponsesClient, _final_text  # noqa: E402

MODEL = "gpt-4o-2024-11-20"
SYSTEM_PATH = ROOT / "prompts/mitigation_extraction_system.txt"
OUT = ROOT / "dataset/mitigation_extraction/extractions.jsonl"


def plan_items(subset, sample=None):
    items = []
    for ds, tasks in subset["tasks"].items():
        for p in load_plan({"source", "tone", "multilingual", "formulation"}, {ds}, set(tasks[:sample] if sample else tasks)):
            if p["form"] in subset["forms"]:
                items.append({"dataset": ds, "task_id": p["task_id"], "family": p["family"], "form": p["form"],
                              "prompt": p["prompt"]})
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key-file", type=Path, default=ROOT / "secrets/openai_key.txt")
    ap.add_argument("--sample", type=int, help="only the first N tasks of each benchmark")
    a = ap.parse_args()
    system = SYSTEM_PATH.read_text(encoding="utf-8")
    system_sha = hashlib.sha256(system.encode()).hexdigest()
    subset = json.loads((ROOT / "dataset/subsets/mitigation_subset.json").read_text())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT.exists():
        for l in OUT.read_text(encoding="utf-8").splitlines():
            r = json.loads(l)
            if r["system_sha256"] == system_sha:
                done.add((r["task_id"], r["form"]))
    todo = [i for i in plan_items(subset, a.sample) if (i["task_id"], i["form"]) not in done]
    print(f"done {len(done)}, to extract {len(todo)}", flush=True)
    client = OpenAIResponsesClient.from_environment(a.api_key_file)
    for i, item in enumerate(todo, 1):
        resp = client.create({"model": MODEL, "instructions": system, "temperature": 0, "max_output_tokens": 2000,
                              "input": [{"role": "user", "content": f"<request>\n{item['prompt']}\n</request>"}],
                              "store": False})
        rec = {**item, "extracted": _final_text(resp.get("output", [])).strip(), "model": resp.get("model"),
               "system_sha256": system_sha, "usage": resp.get("usage")}
        with OUT.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if i % 10 == 0 or i == len(todo):
            print(f"{i}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
