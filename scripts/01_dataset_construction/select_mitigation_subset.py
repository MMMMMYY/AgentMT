#!/usr/bin/env python3
"""10 tasks per benchmark for the mitigation experiment (deterministic): for every benchmark, the first 10
task ids ordered by sha256("mitigation:" + id)."""
import sys as _sys, pathlib as _pl  # scripts are grouped by stage; make sibling stages importable
_sys.path[:0] = [str(_d) for _d in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _d.is_dir()]
import hashlib, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path[:0] = [str(p) for p in sorted((ROOT / "scripts").iterdir()) if p.is_dir()]
from run_full_graph_experiment import load_plan  # noqa: E402
h = lambda salt, t: hashlib.sha256((salt + t).encode()).hexdigest()
tasks = sorted({p["task_id"] for p in load_plan({"source"}, {"AI Agent Permissions", "AgentCIBench", "TRAJECT-Bench"})})
pick = lambda prefix, salt: sorted(sorted([t for t in tasks if t.startswith(prefix)], key=lambda t: h(salt, t))[:10])
out = {"rule": "per benchmark, first 10 task ids by sha256('mitigation:'+id)",
       "tasks": {"AI Agent Permissions": pick("AIP-", "mitigation:"), "AgentCIBench": pick("ACB-", "mitigation:"),
                 "TRAJECT-Bench": pick("TJB-", "mitigation:")},
       "forms": ["source", "p01", "n01", "i01", "lang_zh", "lang_ru", "lang_fr", "lang_es", "formulation"],
       "repeats": 3}
(ROOT / "dataset/subsets/mitigation_subset.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=1))
