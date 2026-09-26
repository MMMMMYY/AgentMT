"""Deterministic operation-graph construction and WL similarity.

The graph contains only observable, policy-relevant agent operations.  Raw
event/call IDs, free-form purpose strings, payload wording, timing, and token
counts are deliberately excluded from semantic labels.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
from typing import Any


AUXILIARY_TOOLS = {
    "list_apps",
    "list_private_fields",
    "list_tools",
    "describe_tool",
    "list_recorded_calls",
    "search_recorded_calls",
    "invoke_recorded_call",
    "invoke_tool",
}

BEHAVIORAL_DIMENSIONS = ("permission", "information", "tool")


def _slug(value: Any) -> str:
    return re.sub(r"[^a-z0-9._:/=-]+", "_", str(value).casefold()).strip("_")


def _stable(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass
class OperationGraph:
    """A directed, edge-labeled graph with discrete semantic node labels."""

    name: str
    nodes: dict[str, str] = field(default_factory=dict)
    edges: list[tuple[str, str, str]] = field(default_factory=list)

    def add_node(self, node_id: str, label: str) -> str:
        existing = self.nodes.get(node_id)
        if existing is not None and existing != label:
            raise ValueError(f"Conflicting labels for {node_id}: {existing} vs {label}")
        self.nodes[node_id] = label
        return node_id

    def entity(self, kind: str, identity: Any) -> str:
        semantic = f"{kind}:{_slug(identity)}"
        return self.add_node(semantic, semantic)

    def add_edge(self, source: str, target: str, relation: str) -> None:
        if source not in self.nodes or target not in self.nodes:
            raise ValueError("Both edge endpoints must exist")
        self.edges.append((source, target, relation))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "semantic_operation_graph_v1",
            "name": self.name,
            "nodes": [
                {"id": node_id, "label": label}
                for node_id, label in sorted(self.nodes.items())
            ],
            "edges": [
                {"source": source, "target": target, "label": relation}
                for source, target, relation in self.edges
            ],
            "excluded_from_labels": [
                "event_id",
                "call_id",
                "free_form_purpose",
                "raw_payload_wording",
                "latency",
                "token_usage",
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OperationGraph":
        graph = cls(payload.get("name", "operation_graph"))
        for node in payload["nodes"]:
            graph.add_node(node["id"], node["label"])
        for edge in payload["edges"]:
            graph.add_edge(edge["source"], edge["target"], edge["label"])
        return graph


def project_behavioral_dimension(graph: OperationGraph, dimension: str) -> OperationGraph:
    """Apply the paper's behavioral projection Pi_k to one operation graph.

    The complete graph is constructed once.  Each projection retains only the
    graph elements relevant to permission, information, or tool behavior while
    preserving semantic tool context and observed order among retained
    operation nodes.
    """
    if dimension not in BEHAVIORAL_DIMENSIONS:
        raise ValueError(f"Unknown behavioral dimension: {dimension}")

    def primary(relation: str) -> bool:
        if dimension == "permission":
            return relation.startswith("REQUESTS:")
        if dimension == "information":
            return relation in {"ACCESSES", "READS", "DISCLOSES_TO", "HAS_OUTPUT"}
        return relation in {"USES", "HAS_ARGUMENT", "CHANGES_STATE"}

    if dimension == "tool":
        selected_operations = {
            node for node, label in graph.nodes.items() if label.startswith("op:")
        }
    else:
        selected_operations = set()
        for source, target, relation in graph.edges:
            if primary(relation):
                if graph.nodes[source].startswith("op:"):
                    selected_operations.add(source)
                if graph.nodes[target].startswith("op:"):
                    selected_operations.add(target)

    kept_edges = []
    for source, target, relation in graph.edges:
        if primary(relation):
            kept_edges.append((source, target, relation))
        elif relation == "USES" and source in selected_operations:
            kept_edges.append((source, target, relation))
        elif relation == "DIRECTLY_PRECEDES" and source in selected_operations and target in selected_operations:
            kept_edges.append((source, target, relation))

    kept_nodes = {endpoint for edge in kept_edges for endpoint in edge[:2]}
    # Tool executions with no outgoing entity edge still remain observable.
    if dimension == "tool":
        kept_nodes.update(selected_operations)
    projected = OperationGraph(f"{graph.name}:{dimension}")
    for node in sorted(kept_nodes):
        projected.add_node(node, graph.nodes[node])
    for source, target, relation in kept_edges:
        projected.add_edge(source, target, relation)
    return projected


def behavioral_similarity(left: OperationGraph, right: OperationGraph, *, h: int = 2) -> dict[str, float]:
    """Normalized WL similarity for every behavioral projection in the paper."""
    return {
        dimension: normalized_wl_similarity(
            project_behavioral_dimension(left, dimension),
            project_behavioral_dimension(right, dimension),
            h=h,
            include_order=True,
        )
        for dimension in BEHAVIORAL_DIMENSIONS
    }


def load_events(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _result_status(result: Any) -> str:
    if isinstance(result, dict):
        if "error" in result:
            return f"error:{_slug(result['error'])}"
        if result.get("status") in {"invalid_request", "invalid_tool_request"}:
            return f"error:{_slug(result['status'])}"
        if result.get("status") == "environment_coverage_gap":
            return "coverage_gap"
    return "success"


def _next_result(events: list[dict[str, Any]], start: int, tool: str) -> str:
    for event in events[start + 1 :]:
        if event["type"] in {"tool_call", "agent_tool_call"}:
            break
        if event["type"] in {"tool_error", "invalid_tool_request"}:
            detail = event.get("result", event)
            return _result_status(detail)
        if event["type"] in {"tool_result", "agent_tool_result"} and event.get("tool") == tool:
            return _result_status(event.get("result"))
    return "unknown"


def _matching_grant(events: list[dict[str, Any]], start: int, request: dict[str, Any]) -> str:
    for event in events[start + 1 :]:
        if event["type"] in {"tool_call", "agent_tool_call"}:
            break
        if event["type"] == "permission_grant" and all(
            event.get(key) == request.get(key) for key in ("operation", "app", "pointer", "destination")
        ):
            return "auto_granted" if event.get("authority") == "sandbox_auto_grant" else "granted"
        if event["type"] == "permission_deny":
            return "denied"
    return "not_granted"


def _add_precedence(graph: OperationGraph, operations: list[str]) -> None:
    for left, right in zip(operations, operations[1:]):
        graph.add_edge(left, right, "DIRECTLY_PRECEDES")


def _agentci_graph(events: list[dict[str, Any]], name: str) -> OperationGraph:
    graph = OperationGraph(name)
    operations: list[str] = []
    active: str | None = None

    for index, event in enumerate(events):
        event_type = event["type"]
        if event_type == "tool_call":
            tool = event["tool"]
            if tool in AUXILIARY_TOOLS:
                active = None
                continue
            status = _next_result(events, index, tool)
            active = graph.add_node(f"op:{len(operations)}", f"op:{_slug(tool)}|outcome:{status}")
            operations.append(active)
            graph.add_edge(active, graph.entity("tool", tool), "USES")
        elif event_type == "permission_request" and active:
            resource = graph.entity("resource", event.get("app", event.get("data", "unknown")))
            decision = _matching_grant(events, index, event)
            relation = (
                f"REQUESTS:{_slug(event.get('operation', 'access'))}"
                f"|scope={_slug(event.get('scope', 'unknown'))}|decision={decision}"
            )
            graph.add_edge(active, resource, relation)
        elif event_type == "access" and active:
            app = event.get("app", event.get("resource", "unknown"))
            graph.add_edge(active, graph.entity("resource", app), "ACCESSES")
            for field_name in event.get("fields", []):
                graph.add_edge(active, graph.entity("field", f"{app}:{field_name}"), "READS")
        elif event_type == "disclosure" and active:
            destination = event.get("destination", event.get("recipient", "unknown"))
            graph.add_edge(active, graph.entity("recipient", destination), "DISCLOSES_TO")
            # Payload wording is evaluated separately and is intentionally not a WL label.
            graph.add_edge(active, graph.entity("output", event.get("operation", "message")), "HAS_OUTPUT")
        elif event_type == "state_change" and active:
            app = event.get("app", "environment")
            destination = event.get("destination", "state")
            graph.add_edge(active, graph.entity("state", f"{app}:{destination}"), "CHANGES_STATE")

    _add_precedence(graph, operations)
    return graph


def _aip_graph(events: list[dict[str, Any]], name: str) -> OperationGraph:
    graph = OperationGraph(name)
    operations: list[str] = []
    active: str | None = None
    for index, event in enumerate(events):
        if event["type"] == "tool_call":
            tool = event["tool"]
            if tool in AUXILIARY_TOOLS:
                active = None
                continue
            status = _next_result(events, index, tool)
            active = graph.add_node(f"op:{len(operations)}", f"op:{_slug(tool)}|outcome:{status}")
            operations.append(active)
            graph.add_edge(active, graph.entity("tool", tool), "USES")
        elif event["type"] == "permission_request" and active:
            field_name = event.get("field", event.get("data", "unknown"))
            scope = _slug(event.get("scope", "unknown"))
            graph.add_edge(active, graph.entity("field", field_name), f"REQUESTS:read|scope={scope}")
    _add_precedence(graph, operations)
    return graph


def _traject_graph(events: list[dict[str, Any]], name: str) -> OperationGraph:
    graph = OperationGraph(name)
    operations: list[str] = []
    for index, event in enumerate(events):
        if event["type"] != "tool_invocation":
            continue
        status = "unknown"
        for later in events[index + 1 :]:
            if later["type"] == "tool_invocation":
                break
            if later["type"] == "fixed_tool_response" and later.get("tool_id") == event.get("tool_id"):
                status = "success"
                break
            if later["type"] == "coverage_gap" and later.get("tool_id") == event.get("tool_id"):
                status = "coverage_gap"
                break
        operation = graph.add_node(
            f"op:{len(operations)}",
            f"op:{_slug(event.get('tool_name', event.get('tool_id', 'tool')))}|outcome:{status}",
        )
        operations.append(operation)
        graph.add_edge(
            operation,
            graph.entity("tool", event.get("tool_name", event.get("tool_id", "tool"))),
            "USES",
        )
        for key, value in sorted(event.get("arguments", {}).items()):
            if isinstance(value, (str, int, float, bool)) or value is None:
                canonical = _slug(_stable(value))
                graph.add_edge(operation, graph.entity("argument", f"{key}={canonical}"), "HAS_ARGUMENT")
    _add_precedence(graph, operations)
    return graph


def build_operation_graph(events: list[dict[str, Any]], *, name: str = "operation_graph") -> OperationGraph:
    types = {event["type"] for event in events}
    if "tool_invocation" in types or "fixed_tool_response" in types or "coverage_gap" in types:
        return _traject_graph(events, name)
    if any(event["type"] == "permission_request" and "field" in event for event in events):
        return _aip_graph(events, name)
    return _agentci_graph(events, name)


def wl_features(graph: OperationGraph, *, h: int = 2, include_order: bool = True) -> Counter[str]:
    """Return edge-aware directed WL subtree features for iterations 0..h."""
    if h < 0:
        raise ValueError("h must be non-negative")
    outgoing: dict[str, list[tuple[str, str]]] = {node: [] for node in graph.nodes}
    incoming: dict[str, list[tuple[str, str]]] = {node: [] for node in graph.nodes}
    for source, target, relation in graph.edges:
        if not include_order and relation == "DIRECTLY_PRECEDES":
            continue
        outgoing[source].append((relation, target))
        incoming[target].append((relation, source))

    labels = dict(graph.nodes)
    features: Counter[str] = Counter(f"0:{label}" for label in labels.values())
    for iteration in range(1, h + 1):
        next_labels: dict[str, str] = {}
        for node in sorted(graph.nodes):
            neighbors = [
                f"OUT|{relation}|{labels[target]}" for relation, target in outgoing[node]
            ] + [
                f"IN|{relation}|{labels[source]}" for relation, source in incoming[node]
            ]
            raw = labels[node] + "||" + "||".join(sorted(neighbors))
            next_labels[node] = _digest(raw)
        labels = next_labels
        features.update(f"{iteration}:{label}" for label in labels.values())
    return features


def normalized_wl_similarity(
    left: OperationGraph,
    right: OperationGraph,
    *,
    h: int = 2,
    include_order: bool = True,
) -> float:
    a = wl_features(left, h=h, include_order=include_order)
    b = wl_features(right, h=h, include_order=include_order)
    dot = sum(value * b.get(key, 0) for key, value in a.items())
    aa = sum(value * value for value in a.values())
    bb = sum(value * value for value in b.values())
    if not aa and not bb:
        return 1.0
    if not aa or not bb:
        return 0.0
    return dot / ((aa * bb) ** 0.5)


def compare_event_logs(left: Path, right: Path, *, h: int = 2) -> dict[str, Any]:
    a = build_operation_graph(load_events(left), name=left.parent.name)
    b = build_operation_graph(load_events(right), name=right.parent.name)
    return {
        "schema": "normalized_wl_operation_similarity_v1",
        "h": h,
        "labels": "fixed_semantic_operation_labels_v1",
        "footprint_similarity": normalized_wl_similarity(a, b, h=h, include_order=False),
        "trajectory_similarity": normalized_wl_similarity(a, b, h=h, include_order=True),
        "left_graph": a.to_dict(),
        "right_graph": b.to_dict(),
    }
