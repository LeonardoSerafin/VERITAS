from __future__ import annotations

import json
import time
from typing import Any

from masfactory import Node


def _is_internal_node_name(name: str) -> bool:
    lowered = name.strip().lower()
    if lowered in {"entry", "exit"}:
        return True
    return lowered.endswith(("_entry", "_exit", "_controller", "_terminate"))


def _to_preview(value: Any, max_len: int) -> str:
    if isinstance(value, (bytes, bytearray)):
        return f"<{type(value).__name__} len={len(value)}>"

    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        text = str(value)

    if len(text) > max_len:
        return text[:max_len] + "...(truncated)"
    return text


def _summarize_output(output: Any, max_len: int) -> str:
    if not isinstance(output, dict):
        return _to_preview(output, max_len=max_len)

    parts: list[str] = []
    for key, value in output.items():
        if key == "image":
            parts.append("image=<omitted>")
        else:
            parts.append(f"{key}={_to_preview(value, max_len=max_len)}")
    return "{ " + ", ".join(parts) + " }"


def install_live_hooks(graph: Any, preview_max_len: int = 220) -> None:
    """
    Attach runtime hooks to print node start/end and node output in real time.
    """

    root_name = str(getattr(graph, "name", "")).strip()
    started_at: dict[int, float] = {}

    def _is_target_node(node: Any) -> bool:
        name = str(getattr(node, "name", "")).strip()
        if not name:
            return False
        if name == root_name:
            return False
        if _is_internal_node_name(name):
            return False
        return True

    def on_execute_before(node: Any, *_args: Any, **_kwargs: Any) -> None:
        if not _is_target_node(node):
            return
        started_at[id(node)] = time.perf_counter()
        print(f"[START] {node.name}")

    def on_forward_after(node: Any, result: Any, *_args: Any, **_kwargs: Any) -> None:
        if not _is_target_node(node):
            return
        print(f"[OUTPUT] {node.name}: {_summarize_output(result, max_len=preview_max_len)}")

    def on_forward_error(node: Any, err: Exception, *_args: Any, **_kwargs: Any) -> None:
        if not _is_target_node(node):
            return
        print(f"[ERROR] {node.name}: {type(err).__name__}: {err}")

    def on_execute_after(node: Any, _result: Any, *_args: Any, **_kwargs: Any) -> None:
        if not _is_target_node(node):
            return
        start = started_at.pop(id(node), None)
        if start is None:
            print(f"[END] {node.name}")
            return
        elapsed_s = (time.perf_counter() - start)
        print(f"[END] {node.name} ({elapsed_s:.1f} s)")

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
    graph.hook_register(
        Node.Hook.EXECUTE.AFTER,
        on_execute_after,
        recursion=True,
        target_type=Node,
        target_filter=_is_target_node,
    )
