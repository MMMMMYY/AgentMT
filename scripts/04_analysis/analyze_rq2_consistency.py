#!/usr/bin/env python3
"""RQ2: is the consequence of an execution the same as under the source prompt?

No correctness judgement: each variant execution is compared only with the source executions.
Order of operations is ignored.  One consequence per benchmark:
  AIP  permission decision : set of requested (data field, scope)
  ACB  information flow    : set of channels of outgoing writes (send_message, calendar, file ...) + set of disclosed items
                             (the benchmark's annotated items are used only as a checklist of what
                             information exists; disclosure per item comes from the GPT-4o judge already run)
  TJB  invocation path     : multiset of invoked tool names (repeated calls count; arguments ignored)
Consistency follows RQ1: inconsistency over the 3x3 source-variant pairs minus inconsistency over
the 3 within-source pairs; task-level mean; task-bootstrap 95% CI.  Reads stored runs only; no API calls.
"""
import sys as _sys, pathlib as _pl  # scripts are grouped by stage; make sibling stages importable
_sys.path[:0] = [str(_d) for _d in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _d.is_dir()]
import csv, itertools, json, random, statistics, sys
from collections import Counter, defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(p) for p in sorted((ROOT / "scripts").iterdir()) if p.is_dir()]; sys.path.insert(0, str(ROOT))
import analyze_rq2_consequences as A
OUT = ROOT / "results/rq2_consequence"
mean = lambda xs: statistics.fmean(xs) if xs else float("nan")


def outcome(a, d):
    ev = A.read_events(d)
    if a["dataset"] == "AI Agent Permissions":
        return frozenset((A.norm(e["field"]), e.get("scope")) for e in ev if e["type"] == "permission_request")
    if a["dataset"] == "AgentCIBench":
        # channel of outgoing writes only: recipient strings vary superficially ("Nora (Manager)" vs
        # "Manager", translated names), so recipient identity is not compared here (routing errors are
        # reported separately in the text)
        writes = frozenset(e.get("operation") for e in ev if e["type"] == "disclosure")
        judged = A.JUDGE.get(str(d.relative_to(ROOT)))
        if judged is not None:
            leaked, missing = judged["leaked_items"], judged["missing_items"]
        else:
            ind = A.indicators(a, d); leaked = ind["leaked_items"]
            disc = A.norm(" \n ".join(str(e.get("payload", "")) for e in ev if e["type"] == "disclosure"))
            missing = [x for x in a["must_share"] if not A.contains(disc, x)]
        shared = {x for x in a["must_share"] if x not in missing} | set(leaked)
        return writes, frozenset(shared)
    return frozenset(Counter(e.get("tool_name") for e in ev if e["type"] == "tool_invocation").items())


ann = A.load_annotations()
CACHE = Path.home() / ".rq2_consistency_cache"; CACHE.mkdir(exist_ok=True)


def canon(x):
    if isinstance(x, (frozenset, set)):
        return sorted((canon(y) for y in x), key=lambda z: json.dumps(z, ensure_ascii=False))
    if isinstance(x, (tuple, list)):
        return [canon(y) for y in x]
    return x


def freeze(o):  # canonical string form; equal outcomes <=> equal strings
    return json.dumps(canon(o), ensure_ascii=False)


runs = defaultdict(list)
only = sys.argv[1] if len(sys.argv) > 1 else None   # optional: one model per call (slow filesystems)
for model, rel in A.CAMPAIGNS:
    cf = CACHE / (model.replace(" ", "_") + ".json")
    if cf.exists():
        cached = json.loads(cf.read_text())
    else:
        if only and only != model:
            continue
        cached = []
        base = ROOT / rel
        for tdir in sorted(p for p in base.iterdir() if p.is_dir() and p.name in ann):
            for fdir in sorted(p for p in tdir.iterdir() if p.is_dir() and "__" in p.name):
                fam, form = fdir.name.split("__", 1)
                for r in ("r0", "r1", "r2"):
                    d = fdir / r
                    try:
                        if json.loads((d / "result.json").read_text()).get("status") == "infrastructure_error":
                            continue
                    except (OSError, json.JSONDecodeError):
                        continue
                    cached.append([tdir.name, fam, form, freeze(outcome(ann[tdir.name], d))])
        cf.write_text(json.dumps(cached))
    for task, fam, form, o in cached:
        runs[(model, task, fam, form)].append(o)
if only:
    print("cached", only); sys.exit(0)

groups = defaultdict(lambda: defaultdict(list))
for (model, task, fam, form), v in runs.items():
    if fam == "source" or not v:
        continue
    s = runs.get((model, task, "source", "source"), [])
    if len(s) < 2:
        continue
    ss = mean([int(x != y) for x, y in itertools.combinations(s, 2)])
    sv = mean([int(x != y) for x in s for y in v])
    groups[(ann[task]["dataset"], model, A.FAMILY[fam])][task].append((ss, sv))

rng = random.Random(0)
def boot(vals, n=2000):
    ms = sorted(mean([rng.choice(vals) for _ in vals]) for _ in range(n)); return ms[int(.025 * n)], ms[int(.975 * n) - 1]
rows = []
for (ds, model, fam), tasks in sorted(groups.items()):
    per = {t: [mean([x[i] for x in lst]) for i in range(2)] for t, lst in tasks.items()}
    delta = [p[1] - p[0] for p in per.values()]
    lo, hi = boot(delta)
    rows.append({"dataset": ds, "model": model, "family": fam, "n_tasks": len(per),
                 "within_source": round(mean([p[0] for p in per.values()]), 4),
                 "source_variant": round(mean([p[1] for p in per.values()]), 4),
                 "delta": round(mean(delta), 4),
                 "se": round(statistics.stdev(delta) / len(delta) ** 0.5, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4)})
with (OUT / "consistency_summary.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows: print(r)
