"""ATIF trajectories (Harbor: agent/trajectory.json) -> semantic operation graphs.

Claude Code and Codex expose different tool sets; every tool call is mapped to a
shared operation vocabulary so that graphs are comparable across executions of
the same agent (the study compares executions within an agent, not across agents).

  file tools   Read/Write/Edit/MultiEdit/NotebookEdit/apply_patch -> read_file / write_file
  search tools Grep/Glob/LS                                         -> search
  shell        Bash / exec_command / shell / local_shell             -> one operation per
               simple command, classified by program: read, write, install, build_run,
               network, vcs, other
  web          WebFetch/WebSearch/web_search                         -> network
  planning     TodoWrite/update_plan/Task                            -> plan (kept as operation)

Labels keep the operation class, the program/tool family and the touched paths;
free-form text (command strings, patch bodies, messages) is excluded, mirroring
the main study's graph labelling.
"""
from __future__ import annotations

import json
import re
import shlex
from pathlib import Path
from typing import Any

from .operation_similarity import OperationGraph

READ = {"cat", "head", "tail", "less", "more", "ls", "find", "grep", "rg", "wc", "file", "stat", "tree", "du",
        "od", "xxd", "hexdump", "strings", "readelf", "objdump", "nm", "diff", "cmp", "which", "whereis", "pwd",
        "echo", "printf", "env", "id", "whoami", "uname", "ps", "df", "awk", "sort", "uniq", "cut", "jq", "type"}
WRITE = {"cp", "mv", "rm", "rmdir", "mkdir", "touch", "chmod", "chown", "tee", "ln", "tar", "unzip", "zip",
         "gzip", "gunzip", "patch", "truncate", "dd", "install"}
INSTALL = {"pip", "pip3", "apt", "apt-get", "conda", "npm", "yarn", "pnpm", "cargo", "uv", "gem", "opam",
           "brew", "dpkg", "apk"}
NETWORK = {"curl", "wget", "ssh", "scp", "nc", "telnet", "ping"}
VCS = {"git"}
PREFIX = {"sudo", "env", "time", "timeout", "nohup", "cd", "export", "source", "."}
FILE_READ = {"read", "view", "notebookread", "read_file"}
FILE_WRITE = {"write", "edit", "multiedit", "notebookedit", "apply_patch", "write_file", "str_replace_editor"}
SEARCH = {"grep", "glob", "ls", "search", "list_dir"}
WEB = {"webfetch", "websearch", "web_search", "fetch"}
PLAN = {"todowrite", "update_plan", "task", "todoread", "exitplanmode"}
SHELL = {"bash", "exec_command", "shell", "local_shell", "container.exec", "run_shell_command", "terminal"}
PATH = re.compile(r"(?:^|[\s'\"=])((?:/|\./|\.\./)?[\w.\-]+(?:/[\w.\-]+)+|[\w\-]+\.[A-Za-z]{1,6})(?=$|[\s'\";:,)])")
ERROR = re.compile(r"(Traceback|No such file|command not found|Permission denied|Error:|error:|FAILED|exit code [1-9])")


def _paths(text: str) -> list[str]:
    out = []
    for m in PATH.finditer(" " + (text or "") + " "):
        p = m.group(1).rstrip(".,:")
        if not re.fullmatch(r"\d+(\.\d+)*", p):
            out.append(p)
    return sorted(set(out))[:12]


def _split_commands(command: str) -> list[list[str]]:
    parts = re.split(r"\s*(?:&&|\|\||;|\|)\s*", command.replace("\n", " ; "))
    cmds = []
    for part in parts:
        if not part.strip():
            continue
        try:
            toks = shlex.split(part, posix=True)
        except ValueError:
            toks = part.split()
        while toks and (toks[0] in PREFIX or re.fullmatch(r"\w+=\S*", toks[0])):
            toks = toks[1:] if toks[0] not in {"cd", "export", "source", "."} else []
        if toks:
            cmds.append(toks)
    return cmds


def classify(toks: list[str]) -> str:
    prog = Path(toks[0]).name
    if prog == "sed" and any(t.startswith("-i") for t in toks):
        return "write"
    if prog in {"python", "python3"} and len(toks) > 2 and toks[1] == "-m" and toks[2] == "pip":
        return "install"
    if prog in READ or prog == "sed":
        return "read"
    if prog in WRITE:
        return "write"
    if prog in INSTALL:
        return "install"
    if prog in NETWORK:
        return "network"
    if prog in VCS:
        return "vcs"
    return "build_run"


def _args(call: dict) -> dict:
    a = call.get("arguments", {})
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except json.JSONDecodeError:
            a = {"command": a}
    return a if isinstance(a, dict) else {"value": a}


def _observations(step: dict) -> dict[str, str]:
    obs = step.get("observation") or {}
    res = obs.get("results", []) if isinstance(obs, dict) else []
    out = {}
    for r in res:
        content = r.get("content")
        out[r.get("source_call_id")] = content if isinstance(content, str) else json.dumps(content)
    return out


def build_graph(trajectory: dict, name: str = "tb2") -> tuple[OperationGraph, list[dict]]:
    g = OperationGraph(name)
    ops, log = [], []

    def add(kind: str, family: str, outcome: str, paths=(), program: str | None = None):
        op = g.add_node(f"op:{len(ops)}", f"op:{kind}|tool:{family}|outcome:{outcome}")
        ops.append(op)
        g.add_edge(op, g.entity("tool", family), "USES")
        if program:
            g.add_edge(op, g.entity("argument", f"program={program}"), "HAS_ARGUMENT")
        rel = {"read": "READS", "read_file": "READS", "search": "READS", "write": "CHANGES_STATE",
               "write_file": "CHANGES_STATE", "network": "DISCLOSES_TO"}.get(kind, "ACCESSES")
        for p in paths:
            g.add_edge(op, g.entity("resource" if kind != "network" else "recipient", p), rel)
        log.append({"kind": kind, "tool": family, "program": program, "paths": list(paths), "outcome": outcome})

    for step in trajectory.get("steps", []):
        if step.get("source") != "agent":
            continue
        obs = _observations(step)
        for call in step.get("tool_calls") or []:
            name_ = str(call.get("function_name", "")).lower()
            args = _args(call)
            text = obs.get(call.get("tool_call_id"), "")
            outcome = "error" if ERROR.search(text or "") else "ok"
            if name_ in SHELL or "command" in args and name_ not in FILE_WRITE:
                cmd = args.get("command") or args.get("cmd") or ""
                if isinstance(cmd, list):
                    cmd = " ".join(map(str, cmd[2:] if cmd[:2] in (["bash", "-lc"], ["bash", "-c"]) else cmd))
                for toks in _split_commands(str(cmd)) or [["<empty>"]]:
                    kind = classify(toks)
                    add(kind, "shell", outcome, _paths(" ".join(toks[1:])), Path(toks[0]).name)
            elif name_ in FILE_READ:
                add("read_file", "file", outcome, _paths(str(args.get("file_path") or args.get("path") or "")))
            elif name_ in FILE_WRITE:
                target = args.get("file_path") or args.get("path") or ""
                body = args.get("input") or args.get("patch") or ""
                files = _paths(str(target)) or sorted(set(re.findall(r"\*\*\* (?:Add|Update|Delete) File: (\S+)", str(body))))
                add("write_file", "file", outcome, files)
            elif name_ in SEARCH:
                add("search", "search", outcome, _paths(str(args.get("path") or "")))
            elif name_ in WEB:
                url = str(args.get("url") or "")
                host = re.sub(r"^https?://([^/]+).*", r"\1", url) if url else "web_search"
                add("network", "web", outcome, [host])
            elif name_ in PLAN:
                add("plan", "plan", "ok")
            else:
                add("other", re.sub(r"\W+", "_", name_) or "unknown", outcome)
    for a, b in zip(ops, ops[1:]):
        g.add_edge(a, b, "DIRECTLY_PRECEDES")
    return g, log


def final_message(trajectory: dict) -> str:
    for step in reversed(trajectory.get("steps", [])):
        if step.get("source") == "agent" and step.get("message") and not step.get("tool_calls"):
            m = step["message"]
            return m if isinstance(m, str) else json.dumps(m, ensure_ascii=False)
    return ""
