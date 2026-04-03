#!/usr/bin/env python3
"""Orchestrate GRN training, smoke tests, and evaluation plots into the results folder."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and test the Causal GRN model, then export publication figures into the results folder."
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--project", required=True, help="Project key inside the YAML config.")
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Directory where run_pipeline.py should write its outputs (training_history.csv, model.pt, ...).",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Directory for evaluation figures/tables. Defaults to results/<project>_<timestamp> next to this script.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Reuse the existing run directory instead of calling run_pipeline.py again.",
    )
    parser.add_argument(
        "--min-epochs",
        type=int,
        default=25,
        help="Minimum epoch count required for the training test to pass.",
    )
    parser.add_argument(
        "--max-loss",
        type=float,
        default=None,
        help="Optional ceiling for the final training loss.",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python interpreter to use when launching sub-commands.",
    )
    parser.add_argument(
        "--extra-train-args",
        nargs="*",
        default=None,
        help="Additional CLI args forwarded to run_pipeline.py (pass after '--extra-train-args ...').",
    )
    return parser.parse_args()


def run_command(cmd: List[str], cwd: Path) -> None:
    print(f"[INFO] Running: {' '.join(cmd)} (cwd={cwd})")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def ensure_scripts(repo_root: Path) -> tuple[Path, Path]:
    pipeline = repo_root / "models" / "causal_grn_project" / "run_pipeline.py"
    evaluator = repo_root / "models" / "causal_grn_project" / "scripts" / "evaluate_grn_model.py"
    if not pipeline.exists():
        raise FileNotFoundError(f"Could not locate run_pipeline.py at {pipeline}")
    if not evaluator.exists():
        raise FileNotFoundError(f"Could not locate evaluate_grn_model.py at {evaluator}")
    return pipeline, evaluator


def run_training(args: argparse.Namespace, pipeline_script: Path) -> None:
    cmd = [
        args.python_bin,
        str(pipeline_script),
        "--config",
        str(args.config),
        "--project",
        args.project,
        "--output-dir",
        str(args.run_dir),
    ]
    if args.extra_train_args:
        cmd.extend(args.extra_train_args)
    run_command(cmd, cwd=pipeline_script.parent)


def run_evaluation(args: argparse.Namespace, evaluator_script: Path, results_dir: Path) -> None:
    cmd = [
        args.python_bin,
        str(evaluator_script),
        "--run-dir",
        str(args.run_dir),
        "--results-dir",
        str(results_dir),
    ]
    run_command(cmd, cwd=evaluator_script.parent)


def list_required_files(run_dir: Path) -> Dict[str, Path]:
    return {
        "training_history": run_dir / "training_history.csv",
        "model_checkpoint": run_dir / "model.pt",
        "edge_table": run_dir / "top_dynamic_edges.csv",
        "condition_map": run_dir / "condition_map.json",
    }


def generate_test_report(run_dir: Path, min_epochs: int, max_loss: float | None) -> Dict[str, object]:
    tests: List[Dict[str, object]] = []
    files = list_required_files(run_dir)
    for name, path in files.items():
        status = "passed" if path.exists() else "failed"
        tests.append({"name": f"{name}_exists", "status": status, "details": str(path)})

    hist_path = files["training_history"]
    if hist_path.exists():
        history = pd.read_csv(hist_path)
        if history.empty or "epoch" not in history.columns or "loss" not in history.columns:
            tests.append(
                {
                    "name": "training_history_valid",
                    "status": "failed",
                    "details": "training_history.csv missing epoch/loss columns or empty",
                }
            )
        else:
            final_row = history.iloc[-1]
            final_epoch = int(final_row["epoch"])
            final_loss = float(final_row["loss"])
            status = "passed" if final_epoch >= min_epochs else "failed"
            tests.append(
                {
                    "name": "min_epoch_requirement",
                    "status": status,
                    "details": f"final_epoch={final_epoch}, required>={min_epochs}",
                }
            )
            if max_loss is not None:
                status = "passed" if final_loss <= max_loss else "failed"
                tests.append(
                    {
                        "name": "max_loss_requirement",
                        "status": status,
                        "details": f"final_loss={final_loss:.4f}, allowed<={max_loss}",
                    }
                )
    else:
        tests.append(
            {
                "name": "training_history_valid",
                "status": "failed",
                "details": "training_history.csv not found",
            }
        )

    return {
        "run_dir": str(run_dir),
        "tests": tests,
        "all_passed": all(t["status"] == "passed" for t in tests),
    }


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    pipeline_script, evaluator_script = ensure_scripts(repo_root)

    run_dir = Path(args.run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    results_root = Path(__file__).resolve().parent
    if args.results_dir:
        results_dir = Path(args.results_dir).expanduser().resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = results_root / f"{args.project}_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_train:
        run_training(args, pipeline_script)
    else:
        print("[INFO] Skipping training; assuming run_dir already populated.")

    run_evaluation(args, evaluator_script, results_dir)
    report = generate_test_report(run_dir, args.min_epochs, args.max_loss)

    report_path = results_dir / "test_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"[INFO] Wrote test report to {report_path}")


if __name__ == "__main__":
    main()
