from __future__ import annotations
import copy
import json
import time
from datetime import datetime, timezone
from typing import Any
from masfactory import Node


TRACKED_NODES = {
    "VisionAgentNode",
    "ContextAgentNode",
    "RAGAgentNode",
    "BypassNode",
    "DecisionAgentNode",
}


class EvaluationRecorder:
    """
    Collects structured data from a VERITAS graph run without affecting graph execution.
    """

    def __init__(self, input_payload: dict[str, Any] | None = None) -> None:
        self.started_at: str | None = None
        self.finished_at: str | None = None
        self.duration_seconds: float | None = None
        self.input_payload = self._sanitize_payload(input_payload or {})
        self.nodes: dict[str, dict[str, Any]] = {}
        self.final_output: Any = None
        self.error: dict[str, Any] | None = None
        self._start_perf: float | None = None
        self._node_start_perf: dict[str, float] = {}

    def start_run(self) -> None:
        self.started_at = datetime.now(timezone.utc).isoformat()
        self._start_perf = time.perf_counter()

    def finish_run(self, final_output: Any = None) -> None:
        self.finished_at = datetime.now(timezone.utc).isoformat()
        if self._start_perf is not None:
            self.duration_seconds = round(time.perf_counter() - self._start_perf, 4)
        self.final_output = self._sanitize_payload(final_output)

    def record_node_start(self, node_name: str) -> None:
        self._node_start_perf[node_name] = time.perf_counter()
        self.nodes.setdefault(node_name, {})
        self.nodes[node_name].update(
            {
                "status": "running",
                "started_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    def record_node_output(self, node_name: str, output: Any) -> None:
        node_record = self.nodes.setdefault(node_name, {})
        started = self._node_start_perf.get(node_name)
        duration = None if started is None else round(time.perf_counter() - started, 4)

        node_record.update(
            {
                "status": "ok",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "duration_seconds": duration,
                "output": self._sanitize_payload(output),
            }
        )

    def record_node_error(self, node_name: str, err: Exception) -> None:
        node_record = self.nodes.setdefault(node_name, {})
        started = self._node_start_perf.get(node_name)
        duration = None if started is None else round(time.perf_counter() - started, 4)
        error = {
            "node": node_name,
            "type": type(err).__name__,
            "message": str(err),
        }

        node_record.update(
            {
                "status": "error",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "duration_seconds": duration,
                "error": error,
            }
        )
        self.error = error

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": self.duration_seconds,
            "input": self.input_payload,
            "nodes": self.nodes,
            "final_output": self.final_output,
            "error": self.error,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def _sanitize_payload(cls, value: Any) -> Any:
        if isinstance(value, dict):
            sanitized = {}
            for key, item in value.items():
                if key == "image":
                    sanitized[key] = cls._summarize_image_asset(item)
                else:
                    sanitized[key] = cls._sanitize_payload(item)
            return sanitized

        if isinstance(value, list):
            return [cls._sanitize_payload(item) for item in value]

        if isinstance(value, tuple):
            return [cls._sanitize_payload(item) for item in value]

        if isinstance(value, (str, int, float, bool)) or value is None:
            return value

        try:
            json.dumps(value, ensure_ascii=False)
            return copy.deepcopy(value)
        except Exception:
            return str(value)

    @staticmethod
    def _summarize_image_asset(value: Any) -> dict[str, Any]:
        return {
            "type": type(value).__name__,
            "source_kind": getattr(value, "source_kind", None),
            "mime_type": getattr(value, "mime_type", None),
            "filename": getattr(value, "filename", None),
            "value": getattr(value, "value", None) if getattr(value, "source_kind", None) == "path" else None,
        }


def install_evaluation_recorder(
    graph: Any,
    recorder: EvaluationRecorder,
    tracked_nodes: set[str] | None = None,
) -> None:
    """
    Attach hooks that record selected node outputs for offline qualitative evaluation.
    """

    tracked = tracked_nodes or TRACKED_NODES

    def _is_target_node(node: Any) -> bool:
        return str(getattr(node, "name", "")).strip() in tracked

    def on_execute_before(node: Any, *_args: Any, **_kwargs: Any) -> None:
        node_name = str(getattr(node, "name", "")).strip()
        recorder.record_node_start(node_name)

    def on_forward_after(node: Any, result: Any, *_args: Any, **_kwargs: Any) -> None:
        node_name = str(getattr(node, "name", "")).strip()
        recorder.record_node_output(node_name, result)

    def on_forward_error(node: Any, err: Exception, *_args: Any, **_kwargs: Any) -> None:
        node_name = str(getattr(node, "name", "")).strip()
        recorder.record_node_error(node_name, err)

    graph.hook_register(
        Node.Hook.EXECUTE.BEFORE,
        on_execute_before,
        recursion=True,
        target_type=Node,
        target_filter=_is_target_node,
    )
    graph.hook_register(
        Node.Hook.FORWARD.AFTER,
        on_forward_after,
        recursion=True,
        target_type=Node,
        target_filter=_is_target_node,
    )
    graph.hook_register(
        Node.Hook.FORWARD.ERROR,
        on_forward_error,
        recursion=True,
        target_type=Node,
        target_filter=_is_target_node,
    )
