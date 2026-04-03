from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn

from .config import ModelConfig
from .io_utils import ProjectTrajectories
from .model import DelayNeuralODEGRN, IntegrationResult


@dataclass
class FitOutput:
    model: DelayNeuralODEGRN
    history: pd.DataFrame
    condition_to_id: Dict[str, int]
    predictions: Dict[str, IntegrationResult]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_observation_tensor(traj: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    obs = np.concatenate([traj["tf_acc"], traj["tf_rna"], traj["target_rna"]], axis=1).astype(np.float32)
    times = np.asarray(traj["time"], dtype=np.float32)
    return torch.tensor(times, device=device), torch.tensor(obs, device=device)


def split_obs(obs: torch.Tensor, n_tfs: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    a = obs[:, :n_tfs]
    r = obs[:, n_tfs : 2 * n_tfs]
    x = obs[:, 2 * n_tfs :]
    return a, r, x


def finite_difference(values: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
    dt = times[1:] - times[:-1]
    dt = dt.clamp_min(1e-4).unsqueeze(1)
    return (values[1:] - values[:-1]) / dt


def compute_loss(
    result: IntegrationResult,
    obs_true: torch.Tensor,
    times: torch.Tensor,
    model: DelayNeuralODEGRN,
    cfg: ModelConfig,
) -> Dict[str, torch.Tensor]:
    pred = result.obs
    a_true, r_true, x_true = split_obs(obs_true, model.n_tfs)
    a_pred, r_pred, x_pred = split_obs(pred, model.n_tfs)

    recon_a = torch.mean((a_pred - a_true) ** 2)
    recon_r = torch.mean((r_pred - r_true) ** 2)
    recon_x = torch.mean((x_pred - x_true) ** 2)
    recon = (
        cfg.recon_weight_tf_acc * recon_a
        + cfg.recon_weight_tf_rna * recon_r
        + cfg.recon_weight_target_rna * recon_x
    )

    if obs_true.shape[0] > 1:
        true_d = finite_difference(obs_true, times)
        pred_d = finite_difference(pred, times)
        deriv_loss = torch.mean((pred_d - true_d) ** 2)
        edge_smooth = torch.mean(torch.abs(result.edges[1:] - result.edges[:-1]))
    else:
        deriv_loss = torch.tensor(0.0, device=obs_true.device)
        edge_smooth = torch.tensor(0.0, device=obs_true.device)

    edge_l1 = torch.mean(torch.abs(result.edges))
    latent_l2 = torch.mean(result.latent ** 2)

    loss = recon
    loss = loss + cfg.deriv_weight * deriv_loss
    loss = loss + cfg.edge_l1 * edge_l1
    loss = loss + cfg.edge_smooth * edge_smooth
    loss = loss + cfg.latent_l2 * latent_l2

    return {
        "loss": loss,
        "recon": recon.detach(),
        "recon_a": recon_a.detach(),
        "recon_r": recon_r.detach(),
        "recon_x": recon_x.detach(),
        "deriv": deriv_loss.detach(),
        "edge_l1": edge_l1.detach(),
        "edge_smooth": edge_smooth.detach(),
        "latent_l2": latent_l2.detach(),
    }


@torch.no_grad()
def infer_all_trajectories(
    model: DelayNeuralODEGRN,
    project_data: ProjectTrajectories,
    condition_to_id: Dict[str, int],
    device: torch.device,
    max_step: float,
) -> Dict[str, IntegrationResult]:
    model.eval()
    outputs: Dict[str, IntegrationResult] = {}
    for traj in project_data.trajectories:
        times, obs = make_observation_tensor(traj, device)
        cond_id = condition_to_id[traj["condition"]]
        result = model.integrate(times, obs[0], cond_id, max_step=max_step)
        outputs[traj["trajectory_id"]] = result
    return outputs


def fit_model(
    project_data: ProjectTrajectories,
    cfg: ModelConfig,
    device: Optional[torch.device] = None,
) -> FitOutput:
    set_seed(cfg.seed)
    device = choose_device() if device is None else device

    condition_to_id = {cond: i for i, cond in enumerate(project_data.conditions)}
    model = DelayNeuralODEGRN(
        n_tfs=len(project_data.tf_names),
        n_targets=len(project_data.target_names),
        n_conditions=len(project_data.conditions),
        latent_dim=cfg.latent_dim,
        hidden_dim=cfg.hidden_dim,
        edge_rank=cfg.edge_rank,
        max_tau=cfg.max_tau,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history_rows: List[Dict[str, float]] = []
    best_loss = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    patience = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = torch.tensor(0.0, device=device)
        metrics_sum: Dict[str, float] = {
            "recon": 0.0,
            "recon_a": 0.0,
            "recon_r": 0.0,
            "recon_x": 0.0,
            "deriv": 0.0,
            "edge_l1": 0.0,
            "edge_smooth": 0.0,
            "latent_l2": 0.0,
        }

        for traj in project_data.trajectories:
            times, obs = make_observation_tensor(traj, device)
            cond_id = condition_to_id[traj["condition"]]
            result = model.integrate(times, obs[0], cond_id, max_step=cfg.max_step)
            loss_dict = compute_loss(result, obs, times, model, cfg)
            total_loss = total_loss + loss_dict["loss"]
            for key in metrics_sum:
                metrics_sum[key] += float(loss_dict[key].cpu().item())

        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        n_traj = max(1, len(project_data.trajectories))
        epoch_loss = float(total_loss.detach().cpu().item()) / n_traj
        row = {"epoch": epoch, "loss": epoch_loss}
        for key, value in metrics_sum.items():
            row[key] = value / n_traj
        taus = model.get_taus()
        row["tau_access_to_tf"] = float(taus["access_to_tf"].detach().cpu().item())
        row["tau_tf_to_target"] = float(taus["tf_to_target"].detach().cpu().item())
        history_rows.append(row)

        if epoch_loss + 1e-6 < best_loss:
            best_loss = epoch_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= cfg.early_stopping_patience:
                break

    if best_state is None:
        raise RuntimeError("Model training failed before producing a valid checkpoint.")
    model.load_state_dict(best_state)

    history_df = pd.DataFrame(history_rows)
    predictions = infer_all_trajectories(model, project_data, condition_to_id, device, cfg.max_step)
    return FitOutput(model=model, history=history_df, condition_to_id=condition_to_id, predictions=predictions)


@torch.no_grad()
def simulate_tf_knockout(
    model: DelayNeuralODEGRN,
    traj: dict,
    condition_id: int,
    tf_index: int,
    device: torch.device,
    start_time: float = 0.5,
    mode: str = "both",
    max_step: float = 0.02,
) -> IntegrationResult:
    times, obs = make_observation_tensor(traj, device)
    return model.integrate(
        times=times,
        obs0=obs[0],
        condition_id=condition_id,
        max_step=max_step,
        intervention={
            "tf_index": int(tf_index),
            "start_time": float(start_time),
            "mode": mode,
        },
    )


def top_edges_dataframe(
    result: IntegrationResult,
    tf_names: List[str],
    target_names: List[str],
    raw_times: np.ndarray,
    condition: str,
    top_k_per_time: int = 200,
) -> pd.DataFrame:
    records: List[dict] = []
    edges = result.edges.detach().cpu().numpy()
    for ti in range(edges.shape[0]):
        w = edges[ti]
        flat_idx = np.argsort(-np.abs(w).ravel())[:top_k_per_time]
        rows, cols = np.unravel_index(flat_idx, w.shape)
        for r, c in zip(rows, cols):
            records.append(
                {
                    "condition": condition,
                    "time_index": ti,
                    "time_value": float(raw_times[ti]),
                    "target": target_names[r],
                    "tf": tf_names[c],
                    "weight": float(w[r, c]),
                    "abs_weight": float(abs(w[r, c])),
                }
            )
    return pd.DataFrame(records)


def save_fit_outputs(
    fit: FitOutput,
    project_data: ProjectTrajectories,
    output_dir: str | Path,
    model_cfg: ModelConfig,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fit.history.to_csv(out / "training_history.csv", index=False)
    with (out / "condition_map.json").open("w", encoding="utf-8") as handle:
        json.dump(fit.condition_to_id, handle, indent=2)

    torch.save(
        {
            "state_dict": fit.model.state_dict(),
            "n_tfs": fit.model.n_tfs,
            "n_targets": fit.model.n_targets,
            "n_conditions": fit.model.n_conditions,
            "latent_dim": fit.model.latent_dim,
            "hidden_dim": fit.model.hidden_dim,
            "edge_rank": fit.model.edge_rank,
            "max_tau": fit.model.max_tau,
            "tf_names": project_data.tf_names,
            "target_names": project_data.target_names,
            "conditions": project_data.conditions,
            "model_config": model_cfg.__dict__,
        },
        out / "model.pt",
    )

    pd.Series(project_data.tf_names).to_csv(out / "selected_tfs.txt", index=False, header=False)
    pd.Series(project_data.target_names).to_csv(out / "selected_targets.txt", index=False, header=False)

    all_edge_tables: List[pd.DataFrame] = []
    for traj in project_data.trajectories:
        result = fit.predictions[traj["trajectory_id"]]
        np.save(out / f"{traj['trajectory_id']}_pred_obs.npy", result.obs.detach().cpu().numpy())
        np.save(out / f"{traj['trajectory_id']}_pred_edges.npy", result.edges.detach().cpu().numpy())
        np.save(out / f"{traj['trajectory_id']}_pred_gates.npy", result.gates.detach().cpu().numpy())
        edge_df = top_edges_dataframe(
            result=result,
            tf_names=project_data.tf_names,
            target_names=project_data.target_names,
            raw_times=np.asarray(traj["raw_time"], dtype=float),
            condition=traj["condition"],
        )
        edge_df["trajectory_id"] = traj["trajectory_id"]
        all_edge_tables.append(edge_df)

    if all_edge_tables:
        pd.concat(all_edge_tables, axis=0).to_csv(out / "top_dynamic_edges.csv", index=False)
