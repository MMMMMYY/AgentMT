#!/usr/bin/env python3
"""RQ3 coding agents on Terminal-Bench 2.0: graph (and, once scored, output) consistency.

Trials are collected from the full job, the replacement-task job and all retry jobs of each agent;
infrastructure failures (classified as in scripts/02_agent_execution/retry_tb2_infra.py) are excluded, agent timeouts kept.
Graphs are built from agent/trajectory.json with privacy_mt.tb2_graph and compared with the RQ1 metric
(normalized WL, h=2, with order): baseline = within-source pairs, variant = source x variant pairs,
variant -> family mean within task -> task mean.
Writes results/rq3_coding/{coverage.json, final_messages.jsonl, graph.csv, output.csv}.
"""
import sys as _sys, pathlib as _pl  # scripts are grouped by stage; make sibling stages importable
_sys.path[:0] = [str(_d) for _d in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _d.is_dir()]
import csv, itertools, json, re, statistics, sys
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path[:0] = [str(p) for p in sorted((ROOT / "scripts").iterdir()) if p.is_dir()]
from privacy_mt.tb2_graph import build_graph, final_message  # noqa: E402
from privacy_mt.operation_similarity import normalized_wl_similarity  # noqa: E402
from retry_tb2_infra import classify, task_form  # noqa: E402

JOBS = ROOT / "runs/terminal_bench"
OUT = ROOT / "results/rq3_coding"
AGENTS = {"Claude Code (Claude Haiku 4.5)": "claude_code_haiku45", "Codex CLI (GPT-6 Luna)": "codex_gpt6luna"}
SELECTED = {t["task"] for t in json.loads((ROOT / "dataset/terminal_bench/selected_tasks.json").read_text())}
FAMILY = {"tone": "Tone", "multilingual": "Language", "formulation": "Formulation"}
mean = lambda xs: statistics.fmean(xs) if xs else float("nan")


def collect(prefix):
    jobs = [p for p in JOBS.iterdir() if p.is_dir() and p.name.startswith(prefix) and "smoke" not in p.name]
    trials = defaultdict(list)
    for job in jobs:
        for t in job.iterdir():
            if not (t / "config.json").exists() or not (t / "agent/trajectory.json").exists():
                continue
            if classify(t) not in ("ok", "agent"):
                continue
            tf = task_form(t)
            if tf.split("__")[0] in SELECTED:
                trials[tf].append(t)
    return {k: sorted(v)[:3] for k, v in trials.items()}   # at most 3 per task-form


def main():
    OUT.mkdir(exist_ok=True)
    coverage, grows, msgs = {}, [], []
    for agent, prefix in AGENTS.items():
        trials = collect(prefix)
        graphs = {}
        for tf, ts in trials.items():
            for t in ts:
                traj = json.loads((t / "agent/trajectory.json").read_text())
                graphs[t] = build_graph(traj, name=t.name)[0]
                msgs.append({"agent": agent, "task_form": tf, "trial": str(t.relative_to(ROOT)),
                             "final_message": final_message(traj)})
        counts = [len(v) for v in trials.values()]
        coverage[agent] = {"task_forms": len(trials), "trials": sum(counts),
                           "forms_with_fewer_than_3": sorted(k for k, v in trials.items() if len(v) < 3)}
        fam_task = defaultdict(dict)
        for task in sorted(SELECTED):
            src = trials.get(f"{task}__source__source", [])
            if len(src) < 2:
                continue
            sg = [graphs[t] for t in src]
            fam_task["Baseline"][task] = [mean([normalized_wl_similarity(a, b, h=2, include_order=True)
                                                for a, b in itertools.combinations(sg, 2)])]
            for tf, ts in trials.items():
                parts = tf.split("__")
                if parts[0] != task or parts[1] == "source":
                    continue
                vg = [graphs[t] for t in ts]
                fam_task[FAMILY[parts[1]]].setdefault(task, []).append(
                    mean([normalized_wl_similarity(a, b, h=2, include_order=True) for a in sg for b in vg]))
        row = {"agent": agent, "n_tasks": len(fam_task["Baseline"])}
        for fam in ["Baseline", "Tone", "Language", "Formulation"]:
            row[fam] = round(mean([mean(v) for v in fam_task[fam].values()]), 3)
        grows.append(row)
    with (OUT / "graph.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(grows[0])); w.writeheader(); w.writerows(grows)
    with (OUT / "final_messages.jsonl").open("w", encoding="utf-8") as f:
        for m in msgs:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    (OUT / "coverage.json").write_text(json.dumps(coverage, indent=1))
    print(json.dumps(coverage, indent=1)); [print(r) for r in grows]


if __name__ == "__main__":
    main()
