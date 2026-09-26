#!/usr/bin/env python3
"""Rerun Terminal-Bench trials that failed for infrastructure reasons (before/around agent start).

Infrastructure failures: AgentSetupTimeoutError, NetworkConnectionError, and NonZeroAgentExitCodeError raised
by the agent *installation* (apt/npm) -- agent timeouts are agent behaviour and are kept.
For every task-form, the number of valid trials is counted across the full job and earlier retry jobs;
failed trial dirs are moved to _failed_infra/<job>/ and the missing attempts are rerun, one small job per
task-form, at low concurrency.

  python3 scripts/02_agent_execution/retry_tb2_infra.py --job claude_code_haiku45_full --agent claude-code \
      --model anthropic/claude-haiku-4-5 --dry-run      # show what would be rerun
  python3 scripts/02_agent_execution/retry_tb2_infra.py ... (without --dry-run) to execute
"""
import sys as _sys, pathlib as _pl  # scripts are grouped by stage; make sibling stages importable
_sys.path[:0] = [str(_d) for _d in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _d.is_dir()]
import argparse, collections, json, os, shutil, subprocess, sys, sysconfig
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
JOBS = ROOT / "runs/terminal_bench"
INFRA = {"AgentSetupTimeoutError", "NetworkConnectionError", "AgentAuthenticationError"}


def classify(trial):
    try:
        r = json.loads((trial / "result.json").read_text())
    except (OSError, json.JSONDecodeError):
        return "running"
    ex = r.get("exception_info") or {}
    t = ex.get("exception_type")
    if not t:
        return "ok"
    if t in INFRA:
        return "infra"
    if t == "NonZeroAgentExitCodeError" and "apt-get" in (ex.get("exception_message") or ""):
        return "infra"
    return "agent"   # e.g. AgentTimeoutError: kept as agent behaviour


def task_form(trial):
    return json.loads((trial / "config.json").read_text())["task"]["path"].rstrip("/").split("/")[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--agent", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--abandon-incomplete", action="store_true",
                    help="treat trials without result.json (e.g. interrupted with Ctrl+C) as infrastructure failures")
    a = ap.parse_args()
    prefix = a.job.rsplit("_full", 1)[0]
    jobs = [JOBS / a.job] + sorted(p for p in JOBS.glob(prefix + "_retry*") if p.is_dir())
    valid, infra = collections.Counter(), []
    for job in jobs:
        for trial in job.iterdir():
            if not trial.is_dir() or trial.name.startswith("_") or not (trial / "config.json").exists():
                continue
            c = classify(trial)
            if c == "running":
                if not a.abandon_incomplete:
                    sys.exit(f"{trial} has no result yet; wait until its job finishes, or pass --abandon-incomplete "
                             "if that job was interrupted")
                infra.append(trial)
                continue
            elif c in ("ok", "agent"):
                valid[task_form(trial)] += 1
            else:
                infra.append(trial)
    # every task-form of the benchmark copy must end up with 3 valid trials (forms whose failed trials were
    # already moved to _failed_infra have no trial left in the job directories)
    forms = sorted(p.name for p in (ROOT / "dataset/terminal_bench/variant_tasks").iterdir() if p.is_dir())
    need = {f: 3 - valid[f] for f in forms if valid[f] < 3}
    print(f"infra-failed trials: {len(infra)}; task-forms needing reruns: {len(need)}; attempts to run: {sum(need.values())}")
    for f, k in sorted(need.items()):
        print(f"  {f}: {k}")
    if a.dry_run:
        return
    for trial in infra:
        dest = JOBS / "_failed_infra" / trial.parent.name / trial.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.move(str(trial), str(dest))
    env = os.environ.copy()   # same key handling as scripts/02_agent_execution/run_tb2_harbor.sh
    for var, fname in (("ANTHROPIC_API_KEY", "anthropic_key.txt"), ("OPENAI_API_KEY", "openai_key.txt")):
        if not env.get(var) and (ROOT / "secrets" / fname).exists():
            env[var] = (ROOT / "secrets" / fname).read_text().strip()
    if a.agent == "claude-code" and not (env.get("ANTHROPIC_API_KEY") or env.get("CLAUDE_CODE_OAUTH_TOKEN")):
        sys.exit("No Anthropic key: set ANTHROPIC_API_KEY or add secrets/anthropic_key.txt")
    harbor = Path(sysconfig.get_path("scripts")) / "harbor"
    harbor = str(harbor) if harbor.exists() else shutil.which("harbor")
    for i, (f, k) in enumerate(sorted(need.items()), 1):
        name = f"{prefix}_retry_infra_{f}"
        print(f"[{i}/{len(need)}] {f} x{k}", flush=True)
        subprocess.run([harbor, "run", "-p", "dataset/terminal_bench/variant_tasks", "-i", f, "-a", a.agent,
                        "-m", a.model, "-k", str(k), "-n", "1", "-o", "runs/terminal_bench",
                        "--job-name", name], cwd=ROOT, check=False, env=env)


if __name__ == "__main__":
    main()
