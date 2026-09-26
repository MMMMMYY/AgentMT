#!/usr/bin/env python3
"""Restore the agent run directories from runs/**/*.jsonl.gz (one JSON object per file: {"path", "content"}).

    python3 scripts/unpack_runs.py          # runs/main/claude-haiku-4.5.jsonl.gz -> runs/main/claude-haiku-4.5/...

Only the files used by the analysis are shipped for every execution: events.jsonl (tool calls, permission
requests, disclosures), operation_graph.json and result.json (final response, status); Terminal-Bench
trials keep config.json, result.json and the ATIF agent/trajectory.json.
"""
import gzip, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for pack in sorted((ROOT / "runs").rglob("*.jsonl.gz")):
    target = pack.with_name(pack.name[: -len(".jsonl.gz")])
    n = 0
    with gzip.open(pack, "rt", encoding="utf-8") as g:
        for line in g:
            rec = json.loads(line)
            out = target / rec["path"]
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(rec["content"], encoding="utf-8")
            n += 1
    print(f"{pack.relative_to(ROOT)}: {n} files -> {target.relative_to(ROOT)}")
