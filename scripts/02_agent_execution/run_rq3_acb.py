#!/usr/bin/env python3
"""RQ3 on the AgentCIBench subset (dataset/subsets/acb_rq3_subset.json): 15 tasks x 9 forms x 3 repeats.

  GPT-6 Astra, single-turn (main-study protocol):
    python3 scripts/02_agent_execution/run_rq3_acb.py --condition single --provider openai --model gpt-6-astra \
        --output runs/rq3_acb/gpt-6-astra --api-key-file secrets/openai_key.txt --max-estimated-cost 80
  Clarification (repeat per provider):
    python3 scripts/02_agent_execution/run_rq3_acb.py --condition clarify --provider anthropic \
        --output runs/rq3_acb/clarify_claude-haiku-4.5 --api-key-file secrets/anthropic_key.txt
Add --limit N for a smoke test.  Re-running resumes; completed episodes are skipped.
"""
import sys as _sys, pathlib as _pl  # scripts are grouped by stage; make sibling stages importable
_sys.path[:0] = [str(_d) for _d in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _d.is_dir()]
import argparse, json, shutil, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue
from threading import Lock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path[:0] = [str(p) for p in sorted((ROOT / "scripts").iterdir()) if p.is_dir()]
from run_full_graph_experiment import load_plan, write_operation_graph, completed_episode  # noqa: E402
from privacy_mt.benchmark_environments import fixture_digest  # noqa: E402
from privacy_mt.fixed_agent_runner import write_json  # noqa: E402
from privacy_mt.rq3_acb import run_episode, SIMULATOR_MODEL  # noqa: E402

PRICES = {  # USD per 1M tokens: input, cached input, output
    "gpt-6-astra": (10.0, 1.0, 50.0), "gpt-6-sol": (2.0, 0.2, 10.0), "gpt-6-luna": (0.10, 0.01, 0.50),
    "gpt-4.1-mini-2025-04-14": (0.40, 0.10, 1.60), SIMULATOR_MODEL: (2.50, 1.25, 10.0)}
DEFAULT_MODEL = {"openai": "gpt-6-astra", "anthropic": "claude-haiku-4-5-20251001", "gemini": "gemini-2.5-flash"}


def openai_cost(model, usages):
    pin, pcache, pout = PRICES.get(model, (0, 0, 0))
    total = 0.0
    for u in usages:
        i = int(u.get("input_tokens", 0) or 0); c = int((u.get("input_tokens_details") or {}).get("cached_tokens", 0) or 0)
        o = int(u.get("output_tokens", 0) or 0)
        total += ((i - c) * pin + c * pcache + o * pout) / 1e6
    return total


def simulator_cost(directory):
    p = directory / "simulator_responses.jsonl"
    return openai_cost(SIMULATOR_MODEL, [json.loads(l).get("usage", {}) for l in p.open()]) if p.exists() else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=["single", "clarify"], required=True)
    ap.add_argument("--provider", choices=["openai", "anthropic", "gemini"], required=True)
    ap.add_argument("--model")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--api-key-file", type=Path)
    ap.add_argument("--sim-key-file", type=Path, default=ROOT / "secrets/openai_key.txt")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, help="run at most N pending episodes (smoke test)")
    ap.add_argument("--only-tasks", nargs="*", help="restrict this start to these task ids (smoke test)")
    ap.add_argument("--only-forms", nargs="*", help="restrict this start to these forms (smoke test)")
    ap.add_argument("--max-estimated-cost", type=float, help="stop scheduling once OpenAI cost (agent+simulator) exceeds this")
    ap.add_argument("--project", default="YOUR_GCP_PROJECT")
    ap.add_argument("--location", default="us-central1")
    a = ap.parse_args()
    a.model = a.model or DEFAULT_MODEL[a.provider]
    a.output = (a.output if a.output.is_absolute() else ROOT / a.output).resolve()
    subset = json.loads((ROOT / "dataset/subsets/acb_rq3_subset.json").read_text())
    families = {"source", "tone", "multilingual", "formulation"}
    plan = [p for p in load_plan(families, {"AgentCIBench"}, set(subset["tasks"])) if p["form"] in subset["forms"]]
    assert len(plan) == len(subset["tasks"]) * len(subset["forms"]), len(plan)
    rows = [{k: v for k, v in p.items() if k not in {"row", "source", "dataset"}} for p in plan]
    manifest = {"schema": "rq3_acb_v1", "condition": a.condition, "provider": a.provider, "model": a.model,
                "subset": subset, "episodes": rows, "plan_sha256": fixture_digest(rows)}
    a.output.mkdir(parents=True, exist_ok=True)
    mpath = a.output / "manifest.json"
    if mpath.exists() and json.loads(mpath.read_text()) != manifest:
        raise SystemExit("Existing manifest differs; choose a new output directory")
    write_json(mpath, manifest)
    pending = []
    for p in plan:
        for r in range(subset["repeats"]):
            d = a.output / p["task_id"] / f'{p["family"]}__{p["form"]}' / f"r{r}"
            if completed_episode(d):
                continue
            if d.exists():
                failed = a.output / "failed_attempts" / p["task_id"] / f'{p["family"]}__{p["form"]}' / f"r{r}_{int(time.time())}"
                failed.parent.mkdir(parents=True, exist_ok=True); shutil.move(str(d), str(failed))
            pending.append((p, d, r))
    total = len(plan) * subset["repeats"]
    print(json.dumps({"planned_runs": total, "remaining": len(pending)}), flush=True)
    if a.only_tasks:
        pending = [e for e in pending if e[0]["task_id"] in a.only_tasks]
    if a.only_forms:
        pending = [e for e in pending if e[0]["form"] in a.only_forms]
    if a.limit is not None:
        pending = pending[:a.limit]

    from privacy_mt.openai_agent_runner import OpenAIResponsesClient
    sim = OpenAIResponsesClient.from_environment(a.sim_key_file) if a.condition == "clarify" else None

    def new_client():
        if a.provider == "openai":
            return OpenAIResponsesClient.from_environment(a.api_key_file)
        if a.provider == "anthropic":
            from privacy_mt.anthropic_agent_runner import AnthropicMessagesClient
            return AnthropicMessagesClient.from_environment(a.api_key_file)
        from privacy_mt.fixed_agent_runner import create_client
        return create_client(a.project, a.location)

    clients, lock, spent = Queue(), Lock(), [0.0]
    for _ in range(min(a.workers, max(1, len(pending)))):
        clients.put(new_client())
    summary = a.output / "run_summary.jsonl"

    def execute(entry):
        p, d, r = entry
        with lock:
            if a.max_estimated_cost is not None and spent[0] >= a.max_estimated_cost:
                return {"task_id": p["task_id"], "form": p["form"], "repeat": r, "status": "estimated_cost_budget"}
        client = clients.get()
        try:
            res = run_episode(a.provider, client, p["row"], p["prompt"], d, model=a.model,
                              condition=a.condition, sim_client=sim)
            write_operation_graph(d)
            cost = (openai_cost(a.model, res.get("usage", [])) if a.provider == "openai" else 0.0) + simulator_cost(d)
            with lock:
                spent[0] += cost
            rec = {"task_id": p["task_id"], "family": p["family"], "form": p["form"], "repeat": r,
                   "status": res["status"], "clarification_rounds": res.get("clarification_rounds"),
                   "estimated_openai_cost_usd": round(cost, 4), "cumulative_usd": round(spent[0], 3)}
        except Exception as exc:
            rec = {"task_id": p["task_id"], "form": p["form"], "repeat": r, "status": "runner_exception",
                   "error_type": type(exc).__name__, "error": str(exc)[:300]}
        finally:
            clients.put(client)
        with lock, summary.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for fut in as_completed([pool.submit(execute, e) for e in pending]):
            print(json.dumps(fut.result(), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
