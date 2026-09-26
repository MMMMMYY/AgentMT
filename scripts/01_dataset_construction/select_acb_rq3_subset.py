#!/usr/bin/env python3
"""Draw the 15 AgentCIBench tasks used by the RQ3 experiments (GPT-6 Astra, clarification).
Deterministic: tasks ordered by sha256("acb-rq3:" + task_id); first 15 kept."""
import sys as _sys, pathlib as _pl  # scripts are grouped by stage; make sibling stages importable
_sys.path[:0] = [str(_d) for _d in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _d.is_dir()]
import hashlib, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
rows = json.loads((ROOT / "dataset/environments/fixed_sources.json").read_text())["agentCiRows"]
tasks = sorted(f"ACB-{r['scenario_id']}" for r in rows)
ranked = sorted(tasks, key=lambda t: hashlib.sha256(("acb-rq3:" + t).encode()).hexdigest())
FORMS = ["source", "p01", "n01", "i01", "lang_zh", "lang_ru", "lang_fr", "lang_es", "formulation"]
out = {"rule": 'first 15 of 36 AgentCIBench tasks ordered by sha256("acb-rq3:" + task_id)',
       "n_pool": len(tasks), "tasks": sorted(ranked[:15]), "forms": FORMS, "repeats": 3}
(ROOT / "dataset/subsets/acb_rq3_subset.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
