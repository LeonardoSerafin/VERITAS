"""
Run VERITAS over a batch of test cases and save one structured JSON per run.

This script is the entry point of the offline evaluation harness. It does NOT
modify the VERITAS graph: it only attaches the EvaluationRecorder hooks to
capture each node's output, then serializes the result to evaluation/runs/.

Usage (from project root):

    .venv/Scripts/python.exe evaluation/run_cases.py
    .venv/Scripts/python.exe evaluation/run_cases.py --cases evaluation/cases/example_cases.jsonl
    .venv/Scripts/python.exe evaluation/run_cases.py --only case_01_black_rot
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Make the project root importable when launched as `python evaluation/run_cases.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings  # noqa: E402
from masfactory import ImageAsset  # noqa: E402
from architecture.masfactory_graph import build_architecture  # noqa: E402
from tools.cnn_leaf_disease_tool import initialize_cnn_tool  # noqa: E402
from evaluation.recorder import (  # noqa: E402
    EvaluationRecorder,
    install_evaluation_recorder,
)

DEFAULT_CASES_FILE = PROJECT_ROOT / "evaluation" / "cases" / "example_cases.jsonl"
DEFAULT_RUNS_DIR = PROJECT_ROOT / "evaluation" / "runs"

# Keys forwarded into the graph payload as the agronomic/meteo context.
CONTEXT_KEYS = ("location", "growth_stage", "wine_type", "recent_treatments")


class _RecorderProxy:
    """
    Thin forwarder so a single set of graph hooks can target a different
    EvaluationRecorder on each case, without rebuilding the graph (which would
    reload the agents' heavy models) and without modifying recorder.py.
    """

    def __init__(self) -> None:
        self.active: EvaluationRecorder | None = None

    def record_node_start(self, node_name: str) -> None:
        if self.active is not None:
            self.active.record_node_start(node_name)

    def record_node_output(self, node_name: str, output: Any) -> None:
        if self.active is not None:
            self.active.record_node_output(node_name, output)

    def record_node_error(self, node_name: str, err: Exception) -> None:
        if self.active is not None:
            self.active.record_node_error(node_name, err)


def load_cases(cases_file: Path) -> list[dict[str, Any]]:
    """Read a JSONL file of cases, skipping blank lines and `#` comments."""
    if not cases_file.exists():
        raise FileNotFoundError(f"Cases file not found: {cases_file}")

    cases: list[dict[str, Any]] = []
    with cases_file.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as err:
                raise ValueError(
                    f"Invalid JSON in {cases_file} at line {line_number}: {err}"
                ) from err
            if "id" not in case:
                raise ValueError(
                    f"Case at line {line_number} in {cases_file} is missing 'id'."
                )
            cases.append(case)
    return cases


def resolve_image_path(image_path: str) -> Path:
    """Resolve a case image path relative to the project root if not absolute."""
    candidate = Path(image_path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate


def build_payload(case: dict[str, Any]) -> dict[str, Any]:
    """Translate a case definition into a VERITAS graph payload."""
    image_path = resolve_image_path(case["image_path"])
    if not image_path.exists():
        raise FileNotFoundError(
            f"Image for case '{case['id']}' not found: {image_path}"
        )

    payload: dict[str, Any] = {
        key: case.get(key, "") for key in CONTEXT_KEYS
    }
    payload["image"] = ImageAsset.from_path(str(image_path))
    return payload


def _mark_skipped_nodes(recorder: EvaluationRecorder) -> None:
    """
    Relabel nodes left in 'running' as 'skipped'.

    The ConditionalNode fires the EXECUTE.BEFORE hook on the branch it does not
    take (so record_node_start runs), but FORWARD.AFTER never fires there. Such
    a node would otherwise stay 'running' with no output, which is misleading
    noise for the downstream evaluator. 'skipped' states plainly that the node
    was not part of the executed path.
    """
    for node_record in recorder.nodes.values():
        if node_record.get("status") == "running":
            node_record["status"] = "skipped"


def run_single_case(
    graph: Any,
    proxy: _RecorderProxy,
    case: dict[str, Any],
) -> EvaluationRecorder:
    """Execute one case and return its populated recorder."""
    payload = build_payload(case)

    recorder = EvaluationRecorder(input_payload=payload)
    proxy.active = recorder
    recorder.start_run()

    result: Any = None
    try:
        result = graph.invoke(payload)
    except Exception as err:  # noqa: BLE001 - we want any failure recorded, not crash the batch
        # Node-level errors are already captured via hooks; record the
        # top-level failure too so the run JSON is self-describing.
        recorder.error = recorder.error or {
            "node": "graph.invoke",
            "type": type(err).__name__,
            "message": str(err),
            "traceback": traceback.format_exc(),
        }
    finally:
        recorder.finish_run(result)
        _mark_skipped_nodes(recorder)
        proxy.active = None

    return recorder


def save_run(
    runs_dir: Path,
    case: dict[str, Any],
    recorder: EvaluationRecorder,
    timestamp: str,
) -> Path:
    """Write a single run JSON, enriched with the originating case metadata."""
    runs_dir.mkdir(parents=True, exist_ok=True)

    record = recorder.to_dict()
    record["case"] = {
        "id": case["id"],
        "description": case.get("description", ""),
        "image_path": case.get("image_path", ""),
    }

    out_path = runs_dir / f"{timestamp}__{case['id']}.json"
    out_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out_path


def run_cases(
    cases_file: Path,
    runs_dir: Path,
    only: str | None = None,
) -> list[Path]:
    """Initialize VERITAS once and run every (selected) case through it."""
    cases = load_cases(cases_file)
    if only:
        cases = [c for c in cases if c["id"] == only]
        if not cases:
            raise ValueError(f"No case with id '{only}' in {cases_file}")

    print(f"Loaded {len(cases)} case(s) from {cases_file}")

    # Heavy, one-time initialization shared across all cases.
    initialize_cnn_tool(
        checkpoint_path=str(settings.CNN_MODEL_PATH),
        data_dir=str(settings.DATASET_DIR),
        image_size=settings.IMAGE_SIZE,
        top_k=settings.VISION_CNN_TOP_K,
    )

    graph = build_architecture()
    proxy = _RecorderProxy()
    install_evaluation_recorder(graph, proxy)

    batch_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    written: list[Path] = []
    for index, case in enumerate(cases, start=1):
        case_id = case["id"]
        print(f"[{index}/{len(cases)}] Running case '{case_id}' ...")
        try:
            recorder = run_single_case(graph, proxy, case)
        except Exception as err:  # noqa: BLE001 - case-setup failure (e.g. missing image)
            print(f"    SKIPPED '{case_id}': {type(err).__name__}: {err}")
            continue

        out_path = save_run(runs_dir, case, recorder, batch_timestamp)
        status = "error" if recorder.error else "ok"
        duration = recorder.duration_seconds
        print(
            f"    {status} in {duration}s -> {out_path.relative_to(PROJECT_ROOT)}"
        )
        written.append(out_path)

    print(f"\nDone. {len(written)} run(s) written to {runs_dir.relative_to(PROJECT_ROOT)}")
    return written


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VERITAS over a batch of evaluation cases."
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES_FILE,
        help=f"Path to the JSONL cases file (default: {DEFAULT_CASES_FILE}).",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help=f"Directory where run JSONs are written (default: {DEFAULT_RUNS_DIR}).",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Run only the case with this id.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_cases(cases_file=args.cases, runs_dir=args.runs_dir, only=args.only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
