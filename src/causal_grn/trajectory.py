from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import ProjectSpec
from .io_utils import LoadedSample, ProjectTrajectories, read_tf_list
from .preprocess import (
    ProjectCellData,
    SelectedFeatures,
    merge_processed_samples,
    preprocess_sample,
    select_features,
    zscore_features,
)
from .pseudotime import (
    build_joint_embedding,
    compute_graph_pseudotime,
    continuous_time_from_discrete_and_pseudotime,
    normalise_time,
)


def _trajectory_group_columns(obs: pd.DataFrame) -> List[str]:
    cols = ["condition"]
    if "replicate" in obs.columns and obs["replicate"].nunique() > 1:
        cols.append("replicate")
    return cols


def _group_name(row: pd.Series, cols: List[str]) -> str:
    return "__".join(str(row[c]) for c in cols)


def _bin_aggregate(
    values: np.ndarray,
    times: np.ndarray,
    n_bins: int,
    min_cells_per_bin: int,
) -> tuple[np.ndarray, np.ndarray]:
    if values.shape[0] != times.shape[0]:
        raise ValueError("values and times must have the same number of rows")
    if values.shape[0] < min_cells_per_bin:
        raise ValueError("Not enough cells to build even a single bin")

    q = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(times, q)
    edges[0] -= 1e-6
    edges[-1] += 1e-6
    bin_ids = np.digitize(times, edges[1:-1], right=False)

    out_values: List[np.ndarray] = []
    out_times: List[float] = []
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() < min_cells_per_bin:
            continue
        out_values.append(values[mask].mean(axis=0))
        out_times.append(float(times[mask].mean()))

    if len(out_values) < 4:
        raise ValueError(
            "Too few populated bins were produced. Lower n_bins or min_cells_per_bin for this trajectory."
        )
    return np.asarray(out_values, dtype=np.float32), np.asarray(out_times, dtype=np.float32)


def _build_embedding_features(data: ProjectCellData, n_var_genes: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    gene_var = data.rna.var(axis=0)
    ranking = np.argsort(-gene_var)
    keep = ranking[: min(n_var_genes, data.rna.shape[1])]
    return data.rna[:, keep].astype(np.float32), data.ga[:, keep].astype(np.float32)


def prepare_project_trajectories(project: ProjectSpec, loaded_samples: List[LoadedSample]) -> ProjectTrajectories:
    processed = [preprocess_sample(s, project.gtf, project.preprocessing) for s in loaded_samples]
    merged = merge_processed_samples(processed, project_name=project.name, species=project.species)

    tf_list = read_tf_list(project.tf_list)
    selected = select_features(merged, project.preprocessing, tf_list=tf_list)

    tf_acc = merged.ga[:, selected.tf_indices].astype(np.float32)
    tf_rna = merged.rna[:, selected.tf_indices].astype(np.float32)
    target_rna = merged.rna[:, selected.target_indices].astype(np.float32)

    tf_acc_z, tf_acc_stats = zscore_features(tf_acc)
    tf_rna_z, tf_rna_stats = zscore_features(tf_rna)
    target_rna_z, target_rna_stats = zscore_features(target_rna)

    rna_emb_source, ga_emb_source = _build_embedding_features(merged)
    joint_embedding = build_joint_embedding(
        rna=rna_emb_source,
        ga=ga_emb_source,
        n_rna_components=project.preprocessing.rna_pcs,
        n_ga_components=project.preprocessing.ga_pcs,
    )

    obs = merged.obs.copy()
    obs["group_cols"] = None
    group_cols = _trajectory_group_columns(obs)
    obs["trajectory_id"] = obs.apply(lambda row: _group_name(row, group_cols), axis=1)

    trajectories: List[dict] = []
    conditions: List[str] = []
    for traj_id, idx in obs.groupby("trajectory_id").indices.items():
        idx_arr = np.asarray(idx, dtype=int)
        traj_obs = obs.iloc[idx_arr].copy()
        actual_time = pd.to_numeric(traj_obs["time"], errors="coerce").to_numpy(dtype=float)
        if np.isnan(actual_time).any():
            raise ValueError(f"Trajectory {traj_id} contains NaN time labels.")
        embedding_sub = joint_embedding[idx_arr]
        pseudotime = compute_graph_pseudotime(
            embedding=embedding_sub,
            actual_time=actual_time,
            n_neighbors=project.preprocessing.n_neighbors,
        )
        continuous_time = continuous_time_from_discrete_and_pseudotime(actual_time, pseudotime)

        tf_acc_bin, bin_time = _bin_aggregate(
            tf_acc_z[idx_arr],
            continuous_time,
            n_bins=project.preprocessing.n_bins,
            min_cells_per_bin=project.preprocessing.min_cells_per_bin,
        )
        tf_rna_bin, _ = _bin_aggregate(
            tf_rna_z[idx_arr],
            continuous_time,
            n_bins=project.preprocessing.n_bins,
            min_cells_per_bin=project.preprocessing.min_cells_per_bin,
        )
        target_bin, _ = _bin_aggregate(
            target_rna_z[idx_arr],
            continuous_time,
            n_bins=project.preprocessing.n_bins,
            min_cells_per_bin=project.preprocessing.min_cells_per_bin,
        )

        cond = str(traj_obs["condition"].iloc[0])
        rep = str(traj_obs["replicate"].iloc[0]) if "replicate" in traj_obs.columns else "rep1"
        conditions.append(cond)
        trajectories.append(
            {
                "trajectory_id": traj_id,
                "condition": cond,
                "replicate": rep,
                "n_cells": int(len(idx_arr)),
                "raw_time": bin_time.astype(np.float32),
                "time": normalise_time(bin_time),
                "tf_acc": tf_acc_bin.astype(np.float32),
                "tf_rna": tf_rna_bin.astype(np.float32),
                "target_rna": target_bin.astype(np.float32),
            }
        )

    feature_stats = {
        "tf_acc_mean": tf_acc_stats["mean"],
        "tf_acc_scale": tf_acc_stats["scale"],
        "tf_rna_mean": tf_rna_stats["mean"],
        "tf_rna_scale": tf_rna_stats["scale"],
        "target_rna_mean": target_rna_stats["mean"],
        "target_rna_scale": target_rna_stats["scale"],
    }
    unique_conditions = sorted(set(conditions))
    return ProjectTrajectories(
        project_name=project.name,
        species=project.species,
        conditions=unique_conditions,
        tf_names=selected.tf_names,
        target_names=selected.target_names,
        trajectories=trajectories,
        feature_stats=feature_stats,
    )
