"""OpenAI Responses API adapter for the fixed benchmark environments.

The model receives one user prompt.  All subsequent inputs are observable tool
results produced by the same local environments used by the Gemini campaign.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import ssl
import time
from typing import Any, Callable
import urllib.error
import urllib.request

import certifi

from .benchmark_environments import fixture_digest
from .fixed_agent_runner import SYSTEM as AGENTCI_SYSTEM, write_json
from .fixed_agentci import FixedAgentCI, event_graph, privacy_projection, tool_declarations
from .aip_permission_agent import (
    SYSTEM as AIP_SYSTEM,
    PermissionPlanningEnvironment,
    declarations as aip_declarations,
    graph as aip_graph,
)
from .traject_fixed_runner import SYSTEM as TRAJECT_SYSTEM, diagnose_reference, task_id
from .traject_response_library import FixedResponseEnvironment, declarations as traject_declarations


DEFAULT_MODEL = "gpt-4.1-mini-2025-04-14"


class OpenAIAPIError(RuntimeError):
    def __init__(self, status_code: int | None, error_code: str | None,
                 parameter: str | None, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.parameter = parameter


def _lower_schema_types(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: (item.casefold() if key == "type" and isinstance(item, str) else _lower_schema_types(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_lower_schema_types(item) for item in value]
    return value


def openai_tools(declarations: list[dict]) -> list[dict]:
    """Convert the shared Gemini-style declarations to Responses API tools."""
    return [
        {
            "type": "function",
            "name": item["name"],
            "description": item.get("description", ""),
            "parameters": _lower_schema_types(item["parameters"]),
            "strict": False,
        }
        for item in declarations
    ]


class OpenAIResponsesClient:
    def __init__(self, api_key: str, *, endpoint: str = "https://api.openai.com/v1/responses"):
        if not api_key.strip():
            raise ValueError("Empty OpenAI API key")
        self._api_key = api_key.strip()
        self.endpoint = endpoint
        self._ssl_context = ssl.create_default_context(cafile=certifi.where())

    @classmethod
    def from_environment(cls, api_key_file: Path | None = None) -> "OpenAIResponsesClient":
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key and api_key_file is not None:
            key = api_key_file.read_text(encoding="utf-8").strip()
        if not key:
            raise RuntimeError("Set OPENAI_API_KEY or provide --api-key-file")
        return cls(key)

    def close(self) -> None:
        self._api_key = ""

    def __enter__(self) -> "OpenAIResponsesClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def create(self, payload: dict, *, attempts: int = 6, timeout: int = 120) -> dict:
        delay = 2
        for attempt in range(attempts):
            request = urllib.request.Request(
                self.endpoint,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={
                    "Authorization": "Bearer " + self._api_key,
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(
                    request, timeout=timeout, context=self._ssl_context
                ) as response:
                    return json.load(response)
            except urllib.error.HTTPError as exc:
                transient = exc.code in {429, 500, 502, 503, 504}
                if not transient or attempt + 1 == attempts:
                    try:
                        body = json.loads(exc.read().decode("utf-8"))
                        error = body.get("error", {}) if isinstance(body, dict) else {}
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        error = {}
                    raise OpenAIAPIError(
                        exc.code,
                        str(error.get("code")) if error.get("code") is not None else None,
                        str(error.get("param")) if error.get("param") is not None else None,
                        str(error.get("message") or f"OpenAI Responses API HTTP {exc.code}"),
                    ) from exc
            except (urllib.error.URLError, TimeoutError, ConnectionResetError,
                    ConnectionAbortedError, BrokenPipeError) as exc:
                if attempt + 1 == attempts:
                    reason = getattr(exc, "reason", exc)
                    raise RuntimeError(
                        "OpenAI Responses API transport failure: "
                        f"{type(reason).__name__}: {str(reason)[:300]}"
                    ) from exc
            time.sleep(delay)
            delay = min(delay * 2, 32)
        raise AssertionError("unreachable")


def _assistant_input_items(output: list[dict]) -> list[dict]:
    """Keep only fields accepted when response output is replayed as input."""
    items = []
    for item in output:
        if item.get("type") == "function_call":
            items.append({key: item[key] for key in ("type", "call_id", "name", "arguments")})
        elif item.get("type") == "message":
            items.append({
                "type": "message",
                "role": item.get("role", "assistant"),
                "content": item.get("content", []),
            })
    return items


def _final_text(output: list[dict]) -> str:
    parts = []
    for item in output:
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                parts.append(content["text"])
    return "\n".join(parts)


def run_tool_loop(
    client: OpenAIResponsesClient,
    *,
    model: str,
    system: str,
    prompt: str,
    declarations: list[dict],
    dispatch: Callable[[str, dict], Any],
    raw_path: Path,
    event_flush: Callable[[], None],
    stop_after_call: Callable[[], str | None] | None = None,
    max_turns: int,
    max_calls: int,
    max_seconds: int,
    max_output_tokens: int,
) -> dict:
    input_items: list[dict] = [{"role": "user", "content": prompt}]
    tools = openai_tools(declarations)
    start = time.monotonic()
    calls = 0
    status, final = "turn_budget", ""
    usage: list[dict] = []
    with raw_path.open("x", encoding="utf-8") as raw:
        for _ in range(max_turns):
            if time.monotonic() - start >= max_seconds:
                status = "time_budget"
                break
            payload = {
                "model": model,
                "instructions": system,
                "input": input_items,
                "tools": tools,
                "tool_choice": "auto",
                "parallel_tool_calls": False,
                "temperature": 0,
                "max_output_tokens": max_output_tokens,
                "store": False,
            }
            response = client.create(payload)
            raw.write(json.dumps(response, ensure_ascii=False) + "\n")
            raw.flush()
            usage.append(response.get("usage", {}))
            output = response.get("output", [])
            function_calls = [item for item in output if item.get("type") == "function_call"]
            input_items.extend(_assistant_input_items(output))
            if not function_calls:
                final = _final_text(output)
                status = "finished_response" if final else ("finished_silent" if calls else "empty_response")
                break
            tool_outputs = []
            for call in function_calls:
                if calls >= max_calls:
                    status = "call_budget"
                    break
                if time.monotonic() - start >= max_seconds:
                    status = "time_budget"
                    break
                try:
                    arguments = json.loads(call.get("arguments") or "{}")
                    if not isinstance(arguments, dict):
                        raise ValueError("Function arguments must be an object")
                    result = dispatch(call["name"], arguments)
                except (json.JSONDecodeError, ValueError, TypeError, KeyError) as exc:
                    result = {"error": "invalid_function_arguments", "error_type": type(exc).__name__}
                calls += 1
                event_flush()
                tool_outputs.append({
                    "type": "function_call_output",
                    "call_id": call["call_id"],
                    "output": json.dumps({"result": result}, ensure_ascii=False),
                })
                stop = stop_after_call() if stop_after_call else None
                if stop:
                    status = stop
                    break
            input_items.extend(tool_outputs)
            if status in {"call_budget", "time_budget", "environment_coverage_gap"}:
                break
    return {
        "status": status,
        "final_response": final,
        "tool_calls": calls,
        "usage": usage,
        "elapsed_seconds": time.monotonic() - start,
    }


def run_agentci_episode(
    client: OpenAIResponsesClient,
    scenario: dict,
    prompt: str,
    output: Path,
    *,
    model: str = DEFAULT_MODEL,
    max_turns: int = 12,
    max_calls: int = 40,
    max_seconds: int = 240,
) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    env = FixedAgentCI(scenario)
    protocol = {
        "system": AGENTCI_SYSTEM,
        "tools": openai_tools(tool_declarations()),
        "model": model,
        "provider": "openai_responses_api",
        "temperature": 0,
        "max_turns": max_turns,
        "max_tool_calls": max_calls,
        "max_seconds": max_seconds,
        "max_output_tokens": 2048,
    }
    write_json(output / "input.json", {
        "source_prompt": scenario["task_prompt"], "prompt": prompt,
        "scenario_id": scenario["scenario_id"], "initial_state": env.state,
        "fixture_sha256": env.initial_digest, "protocol": protocol, "user_messages": 1,
    })
    try:
        result = run_tool_loop(
            client, model=model, system=AGENTCI_SYSTEM, prompt=prompt,
            declarations=tool_declarations(), dispatch=env.dispatch,
            raw_path=output / "raw_responses.jsonl",
            event_flush=lambda: env.save(output / "events.jsonl"),
            max_turns=max_turns, max_calls=max_calls, max_seconds=max_seconds,
            max_output_tokens=2048,
        )
    except Exception as exc:
        env.record("infrastructure_error", error_type=type(exc).__name__,
                   http_status=getattr(exc, "status_code", None),
                   error_code=getattr(exc, "error_code", None),
                   error_parameter=getattr(exc, "parameter", None),
                   error_message=str(exc)[:500])
        result = {"status": "infrastructure_error", "final_response": "", "tool_calls": 0,
                  "usage": [], "elapsed_seconds": 0}
    env.save(output / "events.jsonl")
    write_json(output / "graph.json", event_graph(env.events))
    write_json(output / "projection.json", privacy_projection(env.events))
    write_json(output / "final_state.json", {"apps": env.state, "outbox": env.outbox})
    result.update({
        "diagnostics": env.keyword_diagnostics(), "fixture_sha256": env.initial_digest,
        "provider": "openai_responses_api", "model": model,
        "protocol_sha256": fixture_digest(protocol),
    })
    write_json(output / "result.json", result)
    return result


def run_aip_episode(
    client: OpenAIResponsesClient,
    source: dict,
    prompt: str,
    output: Path,
    *,
    model: str = DEFAULT_MODEL,
) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    env = PermissionPlanningEnvironment(source)
    protocol = {
        "system": AIP_SYSTEM, "tools": openai_tools(aip_declarations()), "model": model,
        "provider": "openai_responses_api", "temperature": 0,
        "max_turns": 8, "max_tool_calls": 30, "max_output_tokens": 1024,
        "private_values_available": False,
    }
    write_json(output / "input.json", {
        "prompt": prompt, "source_prompt": source["query"], "candidate_fields": env.catalog,
        "protocol": protocol, "user_messages": 1,
    })
    try:
        result = run_tool_loop(
            client, model=model, system=AIP_SYSTEM, prompt=prompt,
            declarations=aip_declarations(), dispatch=env.dispatch,
            raw_path=output / "raw_responses.jsonl",
            event_flush=lambda: None,
            max_turns=8, max_calls=30, max_seconds=240, max_output_tokens=1024,
        )
    except Exception as exc:
        env.record("infrastructure_error", error_type=type(exc).__name__,
                   http_status=getattr(exc, "status_code", None),
                   error_code=getattr(exc, "error_code", None),
                   error_parameter=getattr(exc, "parameter", None),
                   error_message=str(exc)[:500])
        result = {"status": "infrastructure_error", "final_response": "", "tool_calls": 0,
                  "usage": [], "elapsed_seconds": 0}
    requests = [
        {"field": event["field"], "scope": event["scope"]}
        for event in env.events if event["type"] == "permission_request"
    ]
    (output / "events.jsonl").write_text("".join(
        json.dumps(event, ensure_ascii=False) + "\n" for event in env.events), encoding="utf-8")
    write_json(output / "graph.json", aip_graph(env.events))
    write_json(output / "projection.json", {"permission_requests": requests})
    result.update({
        "permission_requests": requests, "provider": "openai_responses_api", "model": model,
        "protocol_sha256": fixture_digest(protocol),
    })
    write_json(output / "result.json", result)
    return result


def run_traject_episode(
    client: OpenAIResponsesClient,
    row: dict,
    prompt: str,
    package: dict,
    library_hash: str,
    output: Path,
    *,
    model: str = DEFAULT_MODEL,
    max_turns: int = 64,
    max_calls: int = 160,
    max_seconds: int = 600,
) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    env = FixedResponseEnvironment(package, row["domain"])
    protocol = {
        "system": TRAJECT_SYSTEM, "library_sha256": library_hash, "domain": row["domain"],
        "model": model, "provider": "openai_responses_api", "max_turns": max_turns,
        "max_calls": max_calls, "max_seconds": max_seconds, "temperature": 0,
        "max_output_tokens": 4096, "on_coverage_gap": "stop",
        "tools": openai_tools(traject_declarations()),
    }
    write_json(output / "input.json", {
        "task_id": task_id(row), "source_prompt": row["query"], "prompt": prompt,
        "protocol": protocol, "tool_pool_ids": sorted(env.tools), "user_messages": 1,
    })
    try:
        result = run_tool_loop(
            client, model=model, system=TRAJECT_SYSTEM, prompt=prompt,
            declarations=traject_declarations(), dispatch=env.dispatch,
            raw_path=output / "raw_responses.jsonl",
            event_flush=lambda: env.save(output / "events.jsonl"),
            stop_after_call=lambda: "environment_coverage_gap" if env.gaps else None,
            max_turns=max_turns, max_calls=max_calls, max_seconds=max_seconds,
            max_output_tokens=4096,
        )
    except Exception as exc:
        env.record("infrastructure_error", error_type=type(exc).__name__,
                   http_status=getattr(exc, "status_code", None),
                   error_code=getattr(exc, "error_code", None),
                   error_parameter=getattr(exc, "parameter", None),
                   error_message=str(exc)[:500])
        result = {"status": "infrastructure_error", "final_response": "", "tool_calls": 0,
                  "usage": [], "elapsed_seconds": 0}
    env.save(output / "events.jsonl")
    write_json(output / "graph.json", event_graph(env.events))
    write_json(output / "projection.json", {
        "invocations": [
            {"tool_id": event["tool_id"], "arguments": event["arguments"]}
            for event in env.events if event["type"] == "tool_invocation"
        ],
        "coverage_gaps": env.gaps,
        "note": "Public fixed-response tool use, not evidence of actual private-data access or disclosure",
    })
    result.update({
        "task_success": None,
        "agent_tool_calls_including_discovery": result.pop("tool_calls", 0),
        "functional_invocations_attempted": sum(
            event["type"] == "tool_invocation" for event in env.events
        ),
        "fixed_responses_returned": sum(
            event["type"] == "fixed_tool_response" for event in env.events
        ),
        "coverage_gaps": env.gaps, "censored_by_environment": bool(env.gaps),
        "schema_validation_errors": sum(
            event["type"] == "schema_validation_error" for event in env.events
        ),
        "library_sha256": library_hash, "provider": "openai_responses_api", "model": model,
        "protocol_sha256": fixture_digest(protocol),
        "reference_diagnostics": diagnose_reference(row, package, env.events),
    })
    write_json(output / "result.json", result)
    return result
