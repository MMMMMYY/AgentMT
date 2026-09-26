#!/usr/bin/env python3
"""Random (hash-ranked) 15-task subset of Terminal-Bench 2.0 for RQ3.

Eligibility: difficulty != hard, expert_time_estimate_min <= 60, and the official
agent timeout is the default 900 s (longer timeouts mark long-running tasks).
Ranking: sha256("tb2-rq3:" + task_name); take the first 15.
"""
import sys as _sys, pathlib as _pl  # scripts are grouped by stage; make sibling stages importable
_sys.path[:0] = [str(_d) for _d in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _d.is_dir()]
import hashlib, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
rows = json.loads((ROOT / "dataset/terminal_bench/task_index.json").read_text())
eligible = [r for r in rows if r["difficulty"] != "hard" and (r["expert_min"] or 0) <= 60
            and (r["agent_timeout"] or 0) <= 900]
# Tasks whose environment cannot host the agent harness are replaced by the next
# task in the same hash order (documented, not a judgment about the task itself).
ENVIRONMENT_EXCLUSIONS = {
    "qemu-alpine-ssh": "debian:bullseye-slim image; Debian 11 security packages return 404, so the "
                       "Harbor agent install (apt-get nodejs npm ripgrep) fails before the agent starts (27/27 trials)",
}
eligible = [r for r in eligible if r["task"] not in ENVIRONMENT_EXCLUSIONS]
ranked = sorted(eligible, key=lambda r: hashlib.sha256(f"tb2-rq3:{r['task']}".encode()).hexdigest())
chosen = ranked[:15]
out = [{"task_id": f"TB2-{r['task']}", "task": r["task"], "difficulty": r["difficulty"], "category": r["category"],
        "expert_min": r["expert_min"], "instruction": r["instruction"]} for r in chosen]
(ROOT / "dataset/terminal_bench/selected_tasks.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
print(json.dumps({"total": len(rows), "eligible": len(eligible), "selected": len(out)}))
for r in chosen:
    print(f"{r['task']:<30} {r['difficulty']:<7} {r['category']:<22} {int(r['expert_min'] or 0):>3} min  {r['words']}w")
