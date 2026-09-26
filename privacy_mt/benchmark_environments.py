"""Fail-closed benchmark adapters. Offline replay is NOT live task execution.

Gold labels remain evaluator-side. No missing resource value is synthesized.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any


class EnvironmentUnavailable(RuntimeError):
    pass


class EventLog:
    def __init__(self):
        self.events: list[dict[str, Any]] = []

    def record(self, kind: str, **payload):
        self.events.append({"event_id": len(self.events), "type": kind, **copy.deepcopy(payload)})

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in self.events))


class PermissionEnvironment(EventLog):
    """AIAP data-access gate; completion requires a separate task-specific adapter."""
    def __init__(self, source: dict, fixtures: dict | None = None):
        super().__init__()
        self.source = copy.deepcopy(source)
        self.fixtures = copy.deepcopy(fixtures or {})

    def catalog(self):
        # Only the public candidate schema; never reveal relevance/incorrect labels.
        fields=set(self.source['datatype'])
        for key in ['data_tool1','data_tool2','data_MIRA','data_incorrect',
                    'data_tool1_incorrect','data_tool2_incorrect','data_MIRA_incorrect']:
            fields.update(x.strip() for x in self.source.get(key,'').split(',') if x.strip())
        return sorted(fields)

    def read(self, field: str, scope: str = "once"):
        self.record("permission_request", field=field, scope=scope)
        fixture = self.fixtures.get(field)
        if not fixture or not fixture.get("provenance") or "value" not in fixture:
            self.record("environment_unavailable", field=field)
            raise EnvironmentUnavailable(f"Missing published fixture: {field}")
        self.record("permission_grant", field=field, scope=scope, authority="sandbox_auto_grant")
        self.record("access", field=field, scope=scope, provenance=fixture["provenance"])
        return copy.deepcopy(fixture["value"])


def parameters(tool: dict) -> dict:
    return {p["name"]: p["value"] for group in ("required parameters", "optional parameters")
            for p in tool.get(group, []) if "name" in p and "value" in p}


class TrajectoryReplayEnvironment(EventLog):
    """Exact recorded-call replay; unseen arguments are explicitly unsupported.

    Used for adapter smoke tests, not for measuring unrestricted path expansion.
    The reference tool list must not be exposed as the agent's available tool pool.
    """
    def __init__(self, source: dict):
        super().__init__()
        self.source = copy.deepcopy(source)

    def call(self, name: str, arguments: dict):
        self.record("tool_call", tool=name, arguments=arguments, backend="recorded_replay")
        matches = [t for t in self.source["tool list"]
                   if t["tool name"] == name and parameters(t) == arguments]
        if not matches or any("executed_output" not in t for t in matches):
            self.record("environment_unavailable", reason="unrecorded_call")
            raise EnvironmentUnavailable("No exact published response for this call")
        outputs = [t["executed_output"] for t in matches]
        if any(x != outputs[0] for x in outputs[1:]):
            raise EnvironmentUnavailable("Recorded responses conflict; stateful executor required")
        result = copy.deepcopy(outputs[0])
        self.record("tool_result", tool=name, result=result, backend="recorded_replay")
        return result


def fixture_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
