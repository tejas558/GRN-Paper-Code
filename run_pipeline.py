#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from causal_grn.config import get_project, load_config
from causal_grn.io_utils import load_project_samples
from causal_grn.train import choose_device, fit_model, save_fit_outputs, simulate_tf_knockout
from causal_grn.trajectory import prepare_project_trajectories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a delay-aware dynamic GRN model on single-cell multiome time-series data.")
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    parser.add_argument("--project", required=True, help="Project name from the config file.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional override for the output directory. Defaults to the project's output_dir / project name.",
    )
    parser.add_argument(
        "--perturb-tf",
        action="append",
        default=None,
        help="Optional TF gene name to simulate knockout for after training. Can be passed multiple times.",
    )
    parser.add_argument(
        "--perturb-start",
        type=float,
        default=0.5,
        help="Perturbation start time in normalised trajectory units [0, 1].",
    )
    parser.add_argument(
        "--perturb-mode",
        default="both",
        choices=["acc", "rna", "both"],
        help="Whether the in-silico perturbation zeros TF accessibility, TF RNA, or both.",
    )
    return parser.parse_args()


def save_project_metadata(project_data, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    pd.Series(project_data.tf_names).to_csv(outdir / "selected_tfs.txt", index=False, header=False)
    pd.Series(project_data.target_names).to_csv(outdir / "selected_targets.txt", index=False, header=False)
    np.savez_compressed(outdir / "feature_stats.npz", **project_data.feature_stats)

    traj_rows = []
    for traj in project_data.trajectories:
        traj_rows.append(
            {
                "trajectory_id": traj["trajectory_id"],
                "condition": traj["condition"],
                "replicate": traj["replicate"],
                "n_cells": traj["n_cells"],
                "n_bins": len(traj["time"]),
                "min_raw_time": float(np.min(traj["raw_time"])),
                "max_raw_time": float(np.max(traj["raw_time"])),
            }
        )
        np.save(outdir / f"{traj['trajectory_id']}_times.npy", np.asarray(traj["time"], dtype=np.float32))
        np.save(outdir / f"{traj['trajectory_id']}_raw_times.npy", np.asarray(traj["raw_time"], dtype=np.float32))
        np.save(outdir / f"{traj['trajectory_id']}_tf_acc.npy", np.asarray(traj["tf_acc"], dtype=np.float32))
        np.save(outdir / f"{traj['trajectory_id']}_tf_rna.npy", np.asarray(traj["tf_rna"], dtype=np.float32))
        np.save(outdir / f"{traj['trajectory_id']}_target_rna.npy", np.asarray(traj["target_rna"], dtype=np.float32))
    pd.DataFrame(traj_rows).to_csv(outdir / "trajectory_manifest.csv", index=False)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    project = get_project(cfg, args.project)

    outdir = Path(args.output_dir) if args.output_dir else Path(project.output_dir) / project.name
    outdir.mkdir(parents=True, exist_ok=True)

    loaded_samples = load_project_samples(project)
    project_data = prepare_project_trajectories(project, loaded_samples)
    save_project_metadata(project_data, outdir)

    device = choose_device()
    fit = fit_model(project_data, project.model, device=device)
    save_fit_outputs(fit, project_data, outdir, project.model)

    if args.perturb_tf:
        tf_to_idx = {tf: i for i, tf in enumerate(project_data.tf_names)}
        perturb_dir = outdir / "perturbations"
        perturb_dir.mkdir(parents=True, exist_ok=True)
        for tf_name in args.perturb_tf:
            if tf_name not in tf_to_idx:
                raise KeyError(
                    f"Requested perturbation TF '{tf_name}' is not in the selected TF set. "
                    f"Pick one of the genes in {outdir / 'selected_tfs.txt'}"
                )
            tf_idx = tf_to_idx[tf_name]
            for traj in project_data.trajectories:
                cond_id = fit.condition_to_id[traj["condition"]]
                result = simulate_tf_knockout(
                    model=fit.model,
                    traj=traj,
                    condition_id=cond_id,
                    tf_index=tf_idx,
                    device=device,
                    start_time=args.perturb_start,
                    mode=args.perturb_mode,
                    max_step=project.model.max_step,
                )
                np.save(
                    perturb_dir / f"{traj['trajectory_id']}_{tf_name}_ko_pred_obs.npy",
                    result.obs.detach().cpu().numpy(),
                )
                np.save(
                    perturb_dir / f"{traj['trajectory_id']}_{tf_name}_ko_pred_edges.npy",
                    result.edges.detach().cpu().numpy(),
                )

    summary = {
        "project": project.name,
        "species": project.species,
        "n_trajectories": len(project_data.trajectories),
        "n_tfs": len(project_data.tf_names),
        "n_targets": len(project_data.target_names),
        "output_dir": str(outdir),
    }
    with (outdir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
