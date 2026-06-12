"""
Read saved VERITAS runs and score them with an LLM evaluator.

This is the second half of the offline evaluation harness. It does NOT execute
VERITAS: it consumes the JSON files produced by run_cases.py (in evaluation/runs/),
asks a stronger LLM to grade each one against a state-adaptive rubric
(see prompt.py), and writes one report per run into evaluation/reports/.

Usage (from project root):

    .venv/Scripts/python.exe evaluation/evaluate_outputs.py
    .venv/Scripts/python.exe evaluation/evaluate_outputs.py --runs-dir evaluation/runs
    .venv/Scripts/python.exe evaluation/evaluate_outputs.py --run evaluation/runs/XXX.json
    .venv/Scripts/python.exe evaluation/evaluate_outputs.py --latest   # only the newest batch
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Make the project root importable when launched as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openai import OpenAI  # noqa: E402

from config import settings  # noqa: E402
from evaluation.prompt import (  # noqa: E402
    EVALUATOR_SYSTEM_PROMPT,
    build_evaluation_prompt,
    extract_state,
)

DEFAULT_RUNS_DIR = PROJECT_ROOT / "evaluation" / "runs"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "evaluation" / "reports"


def make_client() -> OpenAI:
    """Build an OpenAI-compatible client pointed at the configured endpoint."""
    if not settings.EVALUATOR_API_KEY:
        raise RuntimeError(
            "EVALUATOR_API_KEY is not set (expected OPENAI_API_KEY in .env)."
        )
    return OpenAI(
        api_key=settings.EVALUATOR_API_KEY,
        base_url=settings.EVALUATOR_BASE_URL,
    )


def _strip_code_fences(text: str) -> str:
    """Remove a ```json ... ``` wrapper if the model added one."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1] if "\n" in cleaned else cleaned
        if cleaned.endswith("```"):
            cleaned = cleaned[: -3]
        # Drop a leading "json" language tag left on its own line.
        if cleaned.lstrip().lower().startswith("json"):
            cleaned = cleaned.lstrip()[4:]
    return cleaned.strip()


def _parse_json_response(content: str) -> dict[str, Any]:
    """Parse the evaluator response into a dict, tolerating minor wrapping."""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    cleaned = _strip_code_fences(content)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Last resort: grab the outermost {...} block.
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


def evaluate_run(client: OpenAI, run: dict[str, Any]) -> dict[str, Any]:
    """Send one run to the evaluator and return the parsed evaluation dict."""
    mode_label, user_prompt = build_evaluation_prompt(run)

    response = client.chat.completions.create(
        model=settings.EVALUATOR_LLM_MODEL_NAME,
        temperature=settings.EVALUATOR_TEMPERATURE,
        messages=[
            {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = response.choices[0].message.content or ""
    evaluation = _parse_json_response(content)
    evaluation["_rubric_mode"] = mode_label
    return evaluation


def build_report(
    run_path: Path,
    run: dict[str, Any],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the on-disk report record for a single run."""
    case = run.get("case", {})
    return {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "evaluator_model": settings.EVALUATOR_LLM_MODEL_NAME,
        "source_run": run_path.name,
        "case_id": case.get("id"),
        "image_state": extract_state(run),
        "run_error": run.get("error"),
        "evaluation": evaluation,
    }


def select_run_files(args: argparse.Namespace) -> list[Path]:
    """Resolve which run JSON files to evaluate from the CLI arguments."""
    if args.run:
        path = Path(args.run)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return [path]

    runs_dir = args.runs_dir
    files = sorted(p for p in runs_dir.glob("*.json"))
    if not files:
        return []

    if args.latest:
        # Batch timestamp is the filename prefix before "__".
        newest_prefix = max(f.name.split("__", 1)[0] for f in files)
        files = [f for f in files if f.name.startswith(newest_prefix)]
    return files


def _overall(evaluation: dict[str, Any]) -> Any:
    return evaluation.get("overall_score", "?")


def run_evaluations(args: argparse.Namespace) -> list[Path]:
    run_files = select_run_files(args)
    if not run_files:
        print(f"No run files found in {args.runs_dir}")
        return []

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    client = make_client()
    print(
        f"Evaluating {len(run_files)} run(s) with "
        f"'{settings.EVALUATOR_LLM_MODEL_NAME}'\n"
    )

    written: list[Path] = []
    summary_rows: list[tuple[str, Any, str]] = []

    for index, run_path in enumerate(run_files, start=1):
        print(f"[{index}/{len(run_files)}] {run_path.name} ...")
        try:
            run = json.loads(run_path.read_text(encoding="utf-8"))
        except Exception as err:  # noqa: BLE001
            print(f"    SKIPPED (cannot read run): {type(err).__name__}: {err}")
            continue

        try:
            evaluation = evaluate_run(client, run)
        except Exception as err:  # noqa: BLE001 - keep grading the rest of the batch
            print(f"    ERROR during evaluation: {type(err).__name__}: {err}")
            summary_rows.append((run.get("case", {}).get("id", run_path.name), "ERR", "-"))
            continue

        report = build_report(run_path, run, evaluation)
        out_path = args.reports_dir / f"{run_path.stem}__eval.json"
        out_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        written.append(out_path)

        overall = _overall(evaluation)
        mode = evaluation.get("_rubric_mode", "-")
        print(f"    overall={overall} [{mode}] -> {out_path.relative_to(PROJECT_ROOT)}")
        summary_rows.append((report.get("case_id") or run_path.name, overall, mode))

    # Final summary table.
    print("\n=== SUMMARY ===")
    for case_id, overall, mode in summary_rows:
        print(f"  {str(case_id):24} overall={overall:<4} [{mode}]")
    print(
        f"\nDone. {len(written)} report(s) written to "
        f"{args.reports_dir.relative_to(PROJECT_ROOT)}"
    )
    return written


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score saved VERITAS runs with an LLM evaluator."
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help=f"Directory with run JSONs (default: {DEFAULT_RUNS_DIR}).",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=DEFAULT_REPORTS_DIR,
        help=f"Directory for evaluation reports (default: {DEFAULT_REPORTS_DIR}).",
    )
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Evaluate only this single run JSON file.",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Evaluate only the most recent batch (by filename timestamp prefix).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_evaluations(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
