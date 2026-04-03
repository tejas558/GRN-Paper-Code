#!/usr/bin/env python3
"""Summarise a trained GRN run with publication-ready figures and tables."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained GRN run and build figures for a manuscript.")
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Directory that contains the outputs from `run_pipeline.py` (training_history.csv, *_pred_obs.npy, etc.).",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Destination for plots/tables. Defaults to results/<run_dir_name> relative to the current working directory.",
    )
    parser.add_argument(
        "--top-features",
        type=int,
        default=4,
        help="Number of genes per block (TF accessibility, TF RNA, target RNA) to visualise in reconstruction plots.",
    )
    parser.add_argument(
        "--top-edges",
        type=int,
        default=12,
        help="Number of TF→target edges per trajectory to include in edge plots.",
    )
    return parser.parse_args()


def _read_gene_list(path: Path) -> Optional[List[str]]:
    if not path.exists():
        print(f"[WARN] Missing gene list: {path}")
        return None
    with path.open("r", encoding="utf-8") as handle:
        genes = [line.strip() for line in handle if line.strip()]
    return genes


def _safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        print(f"[WARN] Missing CSV file: {path}")
        return None
    return pd.read_csv(path)


def _safe_read_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        print(f"[WARN] Missing JSON file: {path}")
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _zscore(series: np.ndarray) -> np.ndarray:
    mean = np.nanmean(series)
    std = np.nanstd(series)
    if std < 1e-6:
        return series - mean
    return (series - mean) / std


class GRNEvaluator:
    def __init__(self, run_dir: Path, results_dir: Path, top_features: int, top_edges: int) -> None:
        self.run_dir = run_dir
        self.results_dir = results_dir
        self.top_features = max(1, int(top_features))
        self.top_edges = max(1, int(top_edges))

        self.figures_dir = self.results_dir / "figures"
        self.tables_dir = self.results_dir / "tables"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)

        sns.set_theme(style="whitegrid", context="talk")

        self.training_history: Optional[pd.DataFrame] = None
        self.trajectory_manifest: Optional[pd.DataFrame] = None
        self.top_edge_table: Optional[pd.DataFrame] = None
        self.run_summary: Optional[Dict] = None
        self.tf_names: Optional[List[str]] = None
        self.target_names: Optional[List[str]] = None
        self.generated_artifacts: List[str] = []

    def run(self) -> None:
        self._load_metadata()
        self._plot_training_history()
        self._plot_delay_traces()
        metrics_df = self._compute_reconstruction_metrics()
        if metrics_df is not None:
            path = self.tables_dir / "reconstruction_metrics.csv"
            metrics_df.to_csv(path, index=False)
            self.generated_artifacts.append(str(path))
        self._plot_trajectory_reconstructions()
        self._plot_edge_heatmaps()
        self._plot_edge_timecourses()
        self._write_summary(metrics_df)

    def _load_metadata(self) -> None:
        self.training_history = _safe_read_csv(self.run_dir / "training_history.csv")
        self.trajectory_manifest = _safe_read_csv(self.run_dir / "trajectory_manifest.csv")
        self.top_edge_table = _safe_read_csv(self.run_dir / "top_dynamic_edges.csv")
        self.run_summary = _safe_read_json(self.run_dir / "run_summary.json")
        self.tf_names = _read_gene_list(self.run_dir / "selected_tfs.txt")
        self.target_names = _read_gene_list(self.run_dir / "selected_targets.txt")

    def _plot_training_history(self) -> None:
        if self.training_history is None or self.training_history.empty:
            return
        hist = self.training_history.copy()
        metric_cols = ["loss", "recon", "deriv", "edge_l1", "edge_smooth", "latent_l2"]
        available = [col for col in metric_cols if col in hist.columns]
        if not available:
            return
        long_df = hist.melt(id_vars="epoch", value_vars=available, var_name="metric", value_name="value")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.lineplot(data=long_df, x="epoch", y="value", hue="metric", ax=ax)
        ax.set_title("Training objective breakdown")
        ax.set_ylabel("Value")
        ax.legend(loc="upper right", frameon=True)
        self._save_figure(fig, self.figures_dir / "training_metrics.png")

    def _plot_delay_traces(self) -> None:
        if self.training_history is None or self.training_history.empty:
            return
        needed = {"tau_access_to_tf", "tau_tf_to_target"}
        if not needed.issubset(set(self.training_history.columns)):
            return
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(self.training_history["epoch"], self.training_history["tau_access_to_tf"], label="Access → TF", lw=2)
        ax.plot(self.training_history["epoch"], self.training_history["tau_tf_to_target"], label="TF → target", lw=2)
        ax.set_title("Learned delays across epochs")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Delay (trajectory units)")
        ax.legend(frameon=True)
        self._save_figure(fig, self.figures_dir / "learned_delays.png")

    def _compute_reconstruction_metrics(self) -> Optional[pd.DataFrame]:
        if self.trajectory_manifest is None or self.trajectory_manifest.empty:
            return None
        if self.tf_names is None or self.target_names is None:
            return None

        records: List[Dict[str, float | str]] = []
        n_tfs = len(self.tf_names)
        for _, row in self.trajectory_manifest.iterrows():
            traj_id = str(row["trajectory_id"])
            blocks = self._load_trajectory_blocks(traj_id, n_tfs)
            if blocks is None:
                continue
            condition = row.get("condition", "")
            for block_name in ("tf_acc", "tf_rna", "target_rna"):
                true_key = f"{block_name}_true"
                pred_key = f"{block_name}_pred"
                true_arr = blocks.get(true_key)
                pred_arr = blocks.get(pred_key)
                if true_arr is None or pred_arr is None:
                    continue
                metrics = self._block_metrics(true_arr, pred_arr)
                record = {
                    "trajectory_id": traj_id,
                    "condition": condition,
                    "block": block_name,
                    **metrics,
                }
                records.append(record)
        if not records:
            return None
        return pd.DataFrame(records)

    @staticmethod
    def _block_metrics(true_arr: np.ndarray, pred_arr: np.ndarray) -> Dict[str, float]:
        diff = pred_arr - true_arr
        mse = float(np.mean(diff**2))
        rmse = float(math.sqrt(max(mse, 0.0)))
        mae = float(np.mean(np.abs(diff)))

        y_true = true_arr.ravel()
        y_pred = pred_arr.ravel()
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        if ss_tot > 1e-8:
            r2 = 1.0 - ss_res / ss_tot
        else:
            r2 = float("nan")
        if np.std(y_true) < 1e-8 or np.std(y_pred) < 1e-8:
            corr = float("nan")
        else:
            corr = float(np.corrcoef(y_true, y_pred)[0, 1])
        return {"rmse": rmse, "mae": mae, "mse": mse, "r2": r2, "pearson_r": corr}

    def _plot_trajectory_reconstructions(self) -> None:
        if self.trajectory_manifest is None or self.trajectory_manifest.empty:
            return
        if self.tf_names is None or self.target_names is None:
            return
        n_tfs = len(self.tf_names)
        for _, row in self.trajectory_manifest.iterrows():
            traj_id = str(row["trajectory_id"])
            condition = row.get("condition", "")
            blocks = self._load_trajectory_blocks(traj_id, n_tfs)
            if blocks is None or blocks.get("times") is None:
                continue
            block_specs = [
                ("TF accessibility", blocks.get("tf_acc_true"), blocks.get("tf_acc_pred"), self.tf_names),
                ("TF RNA", blocks.get("tf_rna_true"), blocks.get("tf_rna_pred"), self.tf_names),
                ("Target RNA", blocks.get("target_rna_true"), blocks.get("target_rna_pred"), self.target_names),
            ]
            valid_blocks = [
                (name, true, pred, genes) for (name, true, pred, genes) in block_specs if true is not None and pred is not None
            ]
            if not valid_blocks:
                continue
            ncols = self.top_features
            nrows = len(valid_blocks)
            fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharex=True)
            axes = np.array(axes, dtype=object)
            if axes.ndim == 1:
                axes = axes.reshape(1, -1)

            for row_idx, (block_name, true_arr, pred_arr, genes) in enumerate(valid_blocks):
                feature_indices = self._select_top_features(true_arr, genes, self.top_features)
                for col_idx in range(ncols):
                    ax = axes[row_idx, col_idx]
                    if col_idx >= len(feature_indices):
                        ax.axis("off")
                        continue
                    feat_idx, label = feature_indices[col_idx]
                    observed = _zscore(true_arr[:, feat_idx])
                    predicted = _zscore(pred_arr[:, feat_idx])
                    ax.plot(blocks["times"], observed, label="Observed", lw=2)
                    ax.plot(blocks["times"], predicted, label="Model", lw=2, linestyle="--")
                    ax.set_title(f"{block_name}: {label}", fontsize=10)
                    if col_idx == 0:
                        ax.set_ylabel("z-score")
                    if row_idx == nrows - 1:
                        ax.set_xlabel("Time")
                    ax.axhline(0, color="gray", lw=0.5, linestyle=":")
                    if row_idx == 0 and col_idx == 0:
                        ax.legend(frameon=True, fontsize=9)

            fig.suptitle(f"{traj_id} ({condition})" if condition else traj_id, fontsize=14)
            self._save_figure(fig, self.figures_dir / f"{traj_id}_reconstructions.png")

    def _plot_edge_heatmaps(self) -> None:
        if self.top_edge_table is None or self.top_edge_table.empty:
            return
        for traj_id, sub in self.top_edge_table.groupby("trajectory_id"):
            pair_scores = (
                sub.groupby(["tf", "target"])["abs_weight"].max().sort_values(ascending=False).head(self.top_edges)
            )
            if pair_scores.empty:
                continue
            selected_pairs = [(tf, target) for tf, target in pair_scores.index]
            labels = [f"{tf} → {target}" for tf, target in selected_pairs]
            working = sub.copy()
            working["pair"] = working["tf"] + " → " + working["target"]
            subset = working[working["pair"].isin(labels)]
            if subset.empty:
                continue
            heatmap = (
                subset.pivot_table(index="pair", columns="time_index", values="weight", aggfunc="mean")
                .reindex(labels)
                .sort_index(axis=1)
            )
            fig, ax = plt.subplots(figsize=(max(6, 0.4 * heatmap.shape[1]), 0.4 * heatmap.shape[0] + 2))
            sns.heatmap(heatmap, cmap="coolwarm", center=0, ax=ax, cbar_kws={"label": "Edge weight"})
            ax.set_title(f"{traj_id} — dynamic TF→target weights")
            ax.set_xlabel("Pseudotime bin")
            ax.set_ylabel("TF → target")
            self._save_figure(fig, self.figures_dir / f"{traj_id}_edge_heatmap.png")

    def _plot_edge_timecourses(self) -> None:
        if self.top_edge_table is None or self.top_edge_table.empty:
            return
        ncols = 3
        for traj_id, sub in self.top_edge_table.groupby("trajectory_id"):
            pair_scores = (
                sub.groupby(["tf", "target"])["abs_weight"].max().sort_values(ascending=False).head(self.top_edges)
            )
            if pair_scores.empty:
                continue
            selected_pairs = [(tf, target) for tf, target in pair_scores.index]
            n_plots = len(selected_pairs)
            nrows = math.ceil(n_plots / ncols)
            fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), sharex=True)
            axes = np.array(axes, dtype=object)
            axes = axes.reshape(nrows, ncols)
            for idx, (tf_name, target_name) in enumerate(selected_pairs):
                ax = axes.flat[idx]
                pair_df = sub[(sub["tf"] == tf_name) & (sub["target"] == target_name)].sort_values("time_value")
                ax.plot(pair_df["time_value"], pair_df["weight"], lw=2)
                ax.axhline(0, color="gray", lw=0.5, linestyle=":")
                ax.set_title(f"{tf_name} → {target_name}", fontsize=10)
                if idx // ncols == nrows - 1:
                    ax.set_xlabel("Time")
                if idx % ncols == 0:
                    ax.set_ylabel("Edge weight")
            for ax in axes.flat[n_plots:]:
                ax.axis("off")
            fig.suptitle(f"{traj_id} — top TF→target edge dynamics", fontsize=14)
            self._save_figure(fig, self.figures_dir / f"{traj_id}_edge_timecourses.png")

    def _load_trajectory_blocks(self, traj_id: str, n_tfs: int) -> Optional[Dict[str, np.ndarray]]:
        pred_path = self.run_dir / f"{traj_id}_pred_obs.npy"
        if not pred_path.exists():
            print(f"[WARN] Missing prediction file: {pred_path}")
            return None
        pred_obs = np.load(pred_path)
        n_targets = pred_obs.shape[1] - 2 * n_tfs
        if n_targets <= 0:
            print(f"[WARN] Prediction tensor for {traj_id} has unexpected shape {pred_obs.shape}")
            return None
        blocks: Dict[str, np.ndarray] = {}
        blocks["tf_acc_pred"] = pred_obs[:, :n_tfs]
        blocks["tf_rna_pred"] = pred_obs[:, n_tfs : 2 * n_tfs]
        blocks["target_rna_pred"] = pred_obs[:, 2 * n_tfs :]

        for suffix in ["tf_acc", "tf_rna", "target_rna", "times", "raw_times"]:
            path = self.run_dir / f"{traj_id}_{suffix}.npy"
            if not path.exists():
                continue
            blocks[f"{suffix}_true" if suffix in {"tf_acc", "tf_rna", "target_rna"} else suffix] = np.load(path)

        if "raw_times" in blocks:
            blocks["times"] = blocks["raw_times"]
        elif "times" not in blocks:
            blocks["times"] = np.arange(pred_obs.shape[0])
        return blocks

    def _select_top_features(
        self, data: np.ndarray, gene_names: Optional[Sequence[str]], top_k: int
    ) -> List[Tuple[int, str]]:
        if data is None or data.size == 0:
            return []
        n_features = data.shape[1]
        k = min(top_k, n_features)
        variances = np.var(data, axis=0)
        order = np.argsort(-variances)[:k]
        labels = []
        for idx in order:
            if gene_names and idx < len(gene_names):
                label = gene_names[idx]
            else:
                label = f"feature_{idx}"
            labels.append((int(idx), label))
        return labels

    def _save_figure(self, fig: plt.Figure, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        self.generated_artifacts.append(str(path))

    def _write_summary(self, metrics_df: Optional[pd.DataFrame]) -> None:
        summary: Dict[str, object] = {}
        if self.run_summary:
            summary.update(self.run_summary)
        if self.training_history is not None and not self.training_history.empty:
            final_row = self.training_history.iloc[-1].to_dict()
            summary["final_training_row"] = final_row
        if metrics_df is not None and not metrics_df.empty:
            summary["aggregate_metrics"] = metrics_df.groupby("block")[["rmse", "mae", "r2", "pearson_r"]].median().to_dict()
        summary["artifacts"] = sorted(self.generated_artifacts)
        path = self.results_dir / "evaluation_summary.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        self.generated_artifacts.append(str(path))


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    if args.results_dir:
        results_dir = Path(args.results_dir).expanduser().resolve()
    else:
        results_dir = Path.cwd() / "results" / run_dir.name
    results_dir.mkdir(parents=True, exist_ok=True)
    evaluator = GRNEvaluator(run_dir, results_dir, top_features=args.top_features, top_edges=args.top_edges)
    evaluator.run()
    print(f"Saved evaluation artifacts to {results_dir}")


if __name__ == "__main__":
    main()
