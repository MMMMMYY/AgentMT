"""Anthropic Messages API adapter for the fixed benchmark environments."""

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


DEFAULT_MODEL = "claude-haiku-4-5-20251001"


class AnthropicAPIError(RuntimeError):
    def __init__(self, status_code: int | None, error_type: str | None, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_type
        self.parameter = None


def _lower_schema_types(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: (item.casefold() if key == "type" and isinstance(item, str)
                  else _lower_schema_types(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_lower_schema_types(item) for item in value]
    return value


def anthropic_tools(declarations: list[dict]) -> list[dict]:
    """Convert the shared Gemini-style declarations to Anthropic tools."""
    return [
        {
            "name": item["name"],
            "description": item.get("description", ""),
            "input_schema": _lower_schema_types(item["parameters"]),
        }
        for item in declarations
    ]


class AnthropicMessagesClient:
    def __init__(self, api_key: str, *, endpoint: str = "https://api.anthropic.com/v1/messages"):
        if not api_key.strip():
            raise ValueError("Empty Anthropic API key")
        self._api_key = api_key.strip()
        self.endpoint = endpoint
        self._ssl_context = ssl.create_default_context(cafile=certifi.where())

    @classmethod
    def from_environment(cls, api_key_file: Path | None = None) -> "AnthropicMessagesClient":
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key and api_key_file is not None:
            key = api_key_file.read_text(encoding="utf-8").strip()
        if not key:
            raise RuntimeError("Set ANTHROPIC_API_KEY or provide --api-key-file")
        return cls(key)

    def close(self) -> None:
        self._api_key = ""

    def __enter__(self) -> "AnthropicMessagesClient":
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
                    "x-api-key": self._api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(
                    request, timeout=timeout, context=self._ssl_context
                ) as response:
                    return json.load(response)
            except urllib.error.HTTPError as exc:
                transient = exc.code in {429, 500, 502, 503, 504, 529}
                try:
                    body = json.loads(exc.read().decode("utf-8"))
                    error = body.get("error", {}) if isinstance(body, dict) else {}
                except (UnicodeDecodeError, json.JSONDecodeError):
                    error = {}
                if not transient or attempt + 1 == attempts:
                    raise AnthropicAPIError(
                        exc.code,
                        str(error.get("type")) if error.get("type") else None,
                        str(error.get("message") or f"Anthropic API HTTP {exc.code}"),
                    ) from exc
            except (urllib.error.URLError, TimeoutError, ConnectionResetError,
                    ConnectionAbortedError, BrokenPipeError) as exc:
                if attempt + 1 == attempts:
                    reason = getattr(exc, "reason", exc)
                    raise RuntimeError(
                        "Anthropic API transport failure: "
                        f"{type(reason).__name__}: {str(reason)[:300]}"
                    ) from exc
            time.sleep(delay)
            delay = min(delay * 2, 32)
        raise AssertionError("unreachable")


def run_tool_loop(
    client: AnthropicMessagesClient,
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
    messages: list[dict] = [{"role": "user", "content": prompt}]
    tools = anthropic_tools(declarations)
    start = time.monotonic()
    calls = 0
    status, final = "turn_budget", ""
    usage: list[dict] = []
    with raw_path.open("x", encoding="utf-8") as raw:
        for _ in range(max_turns):
            if time.monotonic() - start >= max_seconds:
                status = "time_budget"
                break
            response = client.create({
                "model": model,
                "system": system,
                "messages": messages,
                "tools": tools,
                # Automatic prompt caching moves the cache breakpoint forward
                # with the growing multi-turn history.  This changes billing
                # and latency only; the model-visible content is unchanged.
                "cache_control": {"type": "ephemeral"},
                "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
                "temperature": 0,
                "max_tokens": max_output_tokens,
            })
            raw.write(json.dumps(response, ensure_ascii=False) + "\n")
            raw.flush()
            usage.append(response.get("usage", {}))
            content = response.get("content", [])
            messages.append({"role": "assistant", "content": content})
            tool_uses = [block for block in content if block.get("type") == "tool_use"]
            if not tool_uses:
                final = "\n".join(
                    block.get("text", "") for block in content
                    if block.get("type") == "text" and block.get("text")
                )
                status = "finished_response" if final else (
                    "finished_silent" if calls else "empty_response"
                )
                break

            tool_results = []
            budget_hit = False
            for block in tool_uses:
                if calls >= max_calls:
                    status = "call_budget"
                    budget_hit = True
                    break
                if time.monotonic() - start >= max_seconds:
                    status = "time_budget"
                    budget_hit = True
                    break
                arguments = block.get("input", {})
                try:
                    if not isinstance(arguments, dict):
                        raise ValueError("Tool input must be an object")
                    result = dispatch(block["name"], arguments)
                except (ValueError, TypeError, KeyError) as exc:
                    result = {
                        "error": "invalid_function_arguments",
                        "error_type": type(exc).__name__,
                    }
                calls += 1
                event_flush()
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block["id"],
                    "content": json.dumps({"result": result}, ensure_ascii=False),
                })
                if stop_after_call is not None:
                    stop_status = stop_after_call()
                    if stop_status:
                        status = stop_status
                        budget_hit = True
                        break
            if tool_results:
                messages.append({"role": "user", "content": tool_results})
            if budget_hit:
                break
    return {
        "status": status,
        "final_response": final,
        "tool_calls": calls,
        "usage": usage,
        "elapsed_seconds": time.monotonic() - start,
    }


def run_agentci_episode(
    client: AnthropicMessagesClient,
    scenario: dict,
    prompt: str,
    output: Path,
    *,
    model: str = DEFAULT_MODEL,
) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    env = FixedAgentCI(scenario)
    protocol = {
        "system": AGENTCI_SYSTEM,
        "tools": anthropic_tools(tool_declarations()),
        "model": model,
        "provider": "anthropic_messages_api",
        "temperature": 0,
        "max_turns": 12,
        "max_tool_calls": 40,
        "max_seconds": 240,
        "max_output_tokens": 2048,
        "prompt_cache": {"mode": "automatic", "type": "ephemeral", "ttl": "5m"},
    }
    write_json(output / "input.json", {
        "source_prompt": scenario["task_prompt"],
        "prompt": prompt,
        "scenario_id": scenario["scenario_id"],
        "initial_state": env.state,
        "fixture_sha256": env.initial_digest,
        "protocol": protocol,
        "user_messages": 1,
    })
    try:
        result = run_tool_loop(
            client, model=model, system=AGENTCI_SYSTEM, prompt=prompt,
            declarations=tool_declarations(), dispatch=env.dispatch,
            raw_path=output / "raw_responses.jsonl",
            event_flush=lambda: env.save(output / "events.jsonl"),
            max_turns=12, max_calls=40, max_seconds=240, max_output_tokens=2048,
        )
    except Exception as exc:
        env.record(
            "infrastructure_error", error_type=type(exc).__name__,
            http_status=getattr(exc, "status_code", None),
            error_code=getattr(exc, "error_code", None),
            error_parameter=getattr(exc, "parameter", None),
            error_message=str(exc)[:500],
        )
        result = {
            "status": "infrastructure_error", "final_response": "",
            "tool_calls": 0, "usage": [], "elapsed_seconds": 0,
        }
    env.save(output / "events.jsonl")
    write_json(output / "graph.json", event_graph(env.events))
    write_json(output / "projection.json", privacy_projection(env.events))
    write_json(output / "final_state.json", {"apps": env.state, "outbox": env.outbox})
    result.update({
        "diagnostics": env.keyword_diagnostics(),
        "fixture_sha256": env.initial_digest,
        "provider": "anthropic_messages_api",
        "model": model,
        "protocol_sha256": fixture_digest(protocol),
    })
    write_json(output / "result.json", result)
    return result


def run_aip_episode(
    client: AnthropicMessagesClient,
    source: dict,
    prompt: str,
    output: Path,
    *,
    model: str = DEFAULT_MODEL,
) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    env = PermissionPlanningEnvironment(source)
    protocol = {
        "system": AIP_SYSTEM,
        "tools": anthropic_tools(aip_declarations()),
        "model": model,
        "provider": "anthropic_messages_api",
        "temperature": 0,
        "max_turns": 8,
        "max_tool_calls": 30,
        "max_output_tokens": 1024,
        "private_values_available": False,
        "prompt_cache": {"mode": "automatic", "type": "ephemeral", "ttl": "5m"},
    }
    write_json(output / "input.json", {
        "prompt": prompt,
        "source_prompt": source["query"],
        "candidate_fields": env.catalog,
        "protocol": protocol,
        "user_messages": 1,
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
        env.record(
            "infrastructure_error", error_type=type(exc).__name__,
            http_status=getattr(exc, "status_code", None),
            error_code=getattr(exc, "error_code", None),
            error_parameter=getattr(exc, "parameter", None),
            error_message=str(exc)[:500],
        )
        result = {
            "status": "infrastructure_error", "final_response": "",
            "tool_calls": 0, "usage": [], "elapsed_seconds": 0,
        }
    requests = [
        {"field": event["field"], "scope": event["scope"]}
        for event in env.events if event["type"] == "permission_request"
    ]
    (output / "events.jsonl").write_text(
        "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in env.events),
        encoding="utf-8",
    )
    write_json(output / "graph.json", aip_graph(env.events))
    write_json(output / "projection.json", {"permission_requests": requests})
    result.update({
        "permission_requests": requests,
        "provider": "anthropic_messages_api",
        "model": model,
        "protocol_sha256": fixture_digest(protocol),
    })
    write_json(output / "result.json", result)
    return result


def run_traject_episode(
    client: AnthropicMessagesClient,
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
        "system": TRAJECT_SYSTEM,
        "library_sha256": library_hash,
        "domain": row["domain"],
        "model": model,
        "provider": "anthropic_messages_api",
        "max_turns": max_turns,
        "max_calls": max_calls,
        "max_seconds": max_seconds,
        "temperature": 0,
        "max_output_tokens": 4096,
        "on_coverage_gap": "stop",
        "tools": anthropic_tools(traject_declarations()),
        "prompt_cache": {"mode": "automatic", "type": "ephemeral", "ttl": "5m"},
    }
    write_json(output / "input.json", {
        "task_id": task_id(row),
        "source_prompt": row["query"],
        "prompt": prompt,
        "protocol": protocol,
        "tool_pool_ids": sorted(env.tools),
        "user_messages": 1,
    })
    try:
        result = run_tool_loop(
            client,
            model=model,
            system=TRAJECT_SYSTEM,
            prompt=prompt,
            declarations=traject_declarations(),
            dispatch=env.dispatch,
            raw_path=output / "raw_responses.jsonl",
            event_flush=lambda: env.save(output / "events.jsonl"),
            stop_after_call=lambda: "environment_coverage_gap" if env.gaps else None,
            max_turns=max_turns,
            max_calls=max_calls,
            max_seconds=max_seconds,
            max_output_tokens=4096,
        )
    except Exception as exc:
        env.record(
            "infrastructure_error",
            error_type=type(exc).__name__,
            http_status=getattr(exc, "status_code", None),
            error_code=getattr(exc, "error_code", None),
            error_parameter=getattr(exc, "parameter", None),
            error_message=str(exc)[:500],
        )
        result = {
            "status": "infrastructure_error",
            "final_response": "",
            "tool_calls": 0,
            "usage": [],
            "elapsed_seconds": 0,
        }
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
        "coverage_gaps": env.gaps,
        "censored_by_environment": bool(env.gaps),
        "schema_validation_errors": sum(
            event["type"] == "schema_validation_error" for event in env.events
        ),
        "library_sha256": library_hash,
        "provider": "anthropic_messages_api",
        "model": model,
        "protocol_sha256": fixture_digest(protocol),
        "reference_diagnostics": diagnose_reference(row, package, env.events),
    })
    write_json(output / "result.json", result)
    return result
