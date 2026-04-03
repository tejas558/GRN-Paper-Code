from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch import nn


@dataclass
class IntegrationResult:
    obs: torch.Tensor
    latent: torch.Tensor
    edges: torch.Tensor
    gates: torch.Tensor
    taus: Dict[str, float]


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, depth: int = 2, dropout: float = 0.0):
        super().__init__()
        layers: List[nn.Module] = []
        cur = in_dim
        for _ in range(max(1, depth - 1)):
            layers.append(nn.Linear(cur, hidden_dim))
            layers.append(nn.SiLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            cur = hidden_dim
        layers.append(nn.Linear(cur, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DelayNeuralODEGRN(nn.Module):
    def __init__(
        self,
        n_tfs: int,
        n_targets: int,
        n_conditions: int,
        latent_dim: int = 32,
        hidden_dim: int = 128,
        edge_rank: int = 8,
        max_tau: float = 0.35,
    ) -> None:
        super().__init__()
        self.n_tfs = int(n_tfs)
        self.n_targets = int(n_targets)
        self.n_conditions = max(1, int(n_conditions))
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.edge_rank = int(edge_rank)
        self.max_tau = float(max_tau)
        self.obs_dim = 2 * self.n_tfs + self.n_targets
        self.cond_dim = min(16, max(4, hidden_dim // 8))
        self.n_time_freqs = 6
        self.time_dim = 1 + 2 * self.n_time_freqs

        self.encoder = MLP(self.obs_dim, hidden_dim, latent_dim, depth=3)
        self.cond_embedding = nn.Embedding(self.n_conditions, self.cond_dim)

        tf_dyn_in = self.n_tfs + self.n_tfs + latent_dim + self.cond_dim + self.time_dim
        gate_in = self.n_tfs + self.n_tfs + latent_dim + self.cond_dim + self.time_dim
        basal_in = self.n_targets + latent_dim + self.cond_dim + self.time_dim
        latent_in = self.obs_dim + latent_dim + self.cond_dim + self.time_dim
        edge_ctx_in = latent_dim + self.cond_dim + self.time_dim

        self.tf_dynamics = MLP(tf_dyn_in, hidden_dim, 2 * self.n_tfs, depth=3)
        self.gate_net = MLP(gate_in, hidden_dim, self.n_tfs, depth=3)
        self.basal_net = MLP(basal_in, hidden_dim, self.n_targets, depth=3)
        self.latent_dynamics = MLP(latent_in, hidden_dim, latent_dim, depth=3)
        self.edge_context = MLP(edge_ctx_in, hidden_dim, hidden_dim, depth=2)
        self.edge_u = nn.Linear(hidden_dim, self.n_targets * self.edge_rank)
        self.edge_v = nn.Linear(hidden_dim, self.n_tfs * self.edge_rank)

        self.raw_tau_access_to_tf = nn.Parameter(torch.tensor(-2.2))
        self.raw_tau_tf_to_target = nn.Parameter(torch.tensor(-1.8))
        self.raw_decay_obs = nn.Parameter(torch.full((self.obs_dim,), -0.5))
        self.raw_decay_latent = nn.Parameter(torch.full((latent_dim,), -0.5))

        self._init_weights()

    def _init_weights(self) -> None:
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.weight)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)

    def get_taus(self) -> Dict[str, torch.Tensor]:
        tau_a = self.max_tau * torch.sigmoid(self.raw_tau_access_to_tf)
        tau_b = self.max_tau * torch.sigmoid(self.raw_tau_tf_to_target)
        return {
            "access_to_tf": tau_a,
            "tf_to_target": tau_b,
        }

    def encode_initial_state(self, obs0: torch.Tensor) -> torch.Tensor:
        z0 = self.encoder(obs0)
        return torch.cat([obs0, z0], dim=0)

    def _split_state(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        a = state[: self.n_tfs]
        r = state[self.n_tfs : 2 * self.n_tfs]
        x = state[2 * self.n_tfs : self.obs_dim]
        z = state[self.obs_dim :]
        return a, r, x, z

    def _time_features(self, t: torch.Tensor) -> torch.Tensor:
        features = [t.reshape(1)]
        for k in range(self.n_time_freqs):
            freq = (2.0 ** k) * math.pi
            features.append(torch.sin(freq * t).reshape(1))
            features.append(torch.cos(freq * t).reshape(1))
        return torch.cat(features, dim=0)

    def _condition_embedding(self, condition_id: int, device: torch.device) -> torch.Tensor:
        idx = torch.tensor(int(condition_id), dtype=torch.long, device=device)
        return self.cond_embedding(idx)

    def _edge_weights(self, z: torch.Tensor, cond: torch.Tensor, tfeat: torch.Tensor) -> torch.Tensor:
        ctx = torch.cat([z, cond, tfeat], dim=0)
        ctx = self.edge_context(ctx)
        u = self.edge_u(ctx).view(self.n_targets, self.edge_rank)
        v = self.edge_v(ctx).view(self.n_tfs, self.edge_rank)
        w = (u @ v.T) / math.sqrt(float(self.edge_rank))
        return w

    def derivative(
        self,
        t: torch.Tensor,
        state: torch.Tensor,
        lag_access_state: torch.Tensor,
        lag_target_state: torch.Tensor,
        condition_id: int,
    ) -> Dict[str, torch.Tensor]:
        a, r, x, z = self._split_state(state)
        a_lag, _, _, _ = self._split_state(lag_access_state)
        a_target_lag, r_target_lag, _, _ = self._split_state(lag_target_state)
        device = state.device
        cond = self._condition_embedding(condition_id, device=device)
        tfeat = self._time_features(t)

        tf_in = torch.cat([a_lag, r, z, cond, tfeat], dim=0)
        tf_drive = self.tf_dynamics(tf_in)
        da_drive = tf_drive[: self.n_tfs]
        dr_drive = tf_drive[self.n_tfs :]

        gate_in = torch.cat([a_target_lag, r_target_lag, z, cond, tfeat], dim=0)
        gate = torch.sigmoid(self.gate_net(gate_in))

        edge_w = self._edge_weights(z, cond, tfeat)
        basal_in = torch.cat([x, z, cond, tfeat], dim=0)
        basal = self.basal_net(basal_in)

        latent_in = torch.cat([a, r, x, z, cond, tfeat], dim=0)
        dz_drive = self.latent_dynamics(latent_in)

        decay_obs = torch.nn.functional.softplus(self.raw_decay_obs)
        decay_latent = torch.nn.functional.softplus(self.raw_decay_latent)

        da = da_drive - decay_obs[: self.n_tfs] * a
        dr = dr_drive - decay_obs[self.n_tfs : 2 * self.n_tfs] * r
        dx = basal + edge_w @ gate - decay_obs[2 * self.n_tfs :] * x
        dz = dz_drive - decay_latent * z

        dy = torch.cat([da, dr, dx, dz], dim=0)
        return {
            "dy": dy,
            "edge_w": edge_w,
            "gate": gate,
        }

    def _apply_intervention(
        self,
        state: torch.Tensor,
        t_value: float,
        intervention: Optional[Dict[str, float | int | str]],
    ) -> torch.Tensor:
        if intervention is None:
            return state
        start_time = float(intervention.get("start_time", 0.0))
        if t_value < start_time:
            return state
        tf_index = int(intervention["tf_index"])
        mode = str(intervention.get("mode", "both")).lower()
        state = state.clone()
        if mode in {"acc", "both"}:
            state[tf_index] = 0.0
        if mode in {"rna", "both"}:
            state[self.n_tfs + tf_index] = 0.0
        return state

    @staticmethod
    def _interpolate_state(query_t: float, hist_times: List[float], hist_states: List[torch.Tensor]) -> torch.Tensor:
        if query_t <= hist_times[0]:
            return hist_states[0]
        if query_t >= hist_times[-1]:
            return hist_states[-1]
        for i in range(len(hist_times) - 1):
            t0, t1 = hist_times[i], hist_times[i + 1]
            if t0 <= query_t <= t1:
                denom = max(t1 - t0, 1e-8)
                w = float((query_t - t0) / denom)
                return hist_states[i] * (1.0 - w) + hist_states[i + 1] * w
        return hist_states[-1]

    def integrate(
        self,
        times: torch.Tensor,
        obs0: torch.Tensor,
        condition_id: int,
        max_step: float = 0.02,
        intervention: Optional[Dict[str, float | int | str]] = None,
    ) -> IntegrationResult:
        if times.ndim != 1:
            raise ValueError("times must be a 1-D tensor")
        if not torch.all(times[1:] >= times[:-1]):
            raise ValueError("times must be sorted in non-decreasing order")

        state0 = self.encode_initial_state(obs0)
        device = obs0.device
        hist_times: List[float] = [float(times[0].item())]
        hist_states: List[torch.Tensor] = [self._apply_intervention(state0, hist_times[0], intervention)]

        obs_records: List[torch.Tensor] = [hist_states[0][: self.obs_dim]]
        latent_records: List[torch.Tensor] = [hist_states[0][self.obs_dim :]]

        tau_vals = self.get_taus()
        init_aux = self.derivative(
            times[0],
            hist_states[0],
            hist_states[0],
            hist_states[0],
            condition_id=condition_id,
        )
        edge_records: List[torch.Tensor] = [init_aux["edge_w"]]
        gate_records: List[torch.Tensor] = [init_aux["gate"]]

        for j in range(len(times) - 1):
            t_start = float(times[j].item())
            t_end = float(times[j + 1].item())
            interval = max(t_end - t_start, 1e-8)
            n_steps = max(1, int(math.ceil(interval / max_step)))
            h = interval / n_steps

            current_state = hist_states[-1]
            current_t = hist_times[-1]
            aux_out = None
            for _ in range(n_steps):
                tau_vals = self.get_taus()
                lag_access = self._interpolate_state(
                    current_t - float(tau_vals["access_to_tf"].detach().item()), hist_times, hist_states
                )
                lag_target = self._interpolate_state(
                    current_t - float(tau_vals["tf_to_target"].detach().item()), hist_times, hist_states
                )
                t_tensor = torch.tensor(current_t, dtype=times.dtype, device=device)
                aux = self.derivative(
                    t=t_tensor,
                    state=current_state,
                    lag_access_state=lag_access,
                    lag_target_state=lag_target,
                    condition_id=condition_id,
                )
                next_state = current_state + h * aux["dy"]
                next_t = current_t + h
                next_state = self._apply_intervention(next_state, next_t, intervention)
                hist_times.append(next_t)
                hist_states.append(next_state)
                current_state = next_state
                current_t = next_t
                aux_out = aux

            if aux_out is None:
                raise RuntimeError("Integration produced no steps.")
            obs_records.append(current_state[: self.obs_dim])
            latent_records.append(current_state[self.obs_dim :])
            edge_records.append(aux_out["edge_w"])
            gate_records.append(aux_out["gate"])

        return IntegrationResult(
            obs=torch.stack(obs_records, dim=0),
            latent=torch.stack(latent_records, dim=0),
            edges=torch.stack(edge_records, dim=0),
            gates=torch.stack(gate_records, dim=0),
            taus={k: float(v.detach().cpu().item()) for k, v in tau_vals.items()},
        )
