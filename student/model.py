"""Koopman-style polynomial world model.

This branch uses fixed nonlinear lifting features and a linear controlled
latent transition instead of a generic MLP. The public interface stays locked
to the homework harness.
"""

from __future__ import annotations

import torch
from torch import nn


class StudentWorldModel(nn.Module):
    def __init__(
        self,
        obs_dim: int = 4,
        act_dim: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 2,
        use_gru: bool = False,
        delta_limit: float = 3.0,
    ):
        super().__init__()
        self.use_gru = False
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.delta_limit = float(delta_limit)
        latent_dim = max(8, int(hidden_dim))

        state_feature_dim = 1 + (4 * self.obs_dim) + (self.obs_dim * (self.obs_dim - 1) // 2)
        control_feature_dim = self.act_dim + self.act_dim + (self.obs_dim * self.act_dim)
        self.state_lift = nn.Linear(state_feature_dim, latent_dim, bias=False)
        self.control_lift = nn.Linear(control_feature_dim, latent_dim, bias=False)
        self.transition = nn.Linear(latent_dim, latent_dim, bias=False)
        self.decoder = nn.Linear(latent_dim, obs_dim)

        nn.init.eye_(self.transition.weight)
        nn.init.xavier_uniform_(self.state_lift.weight, gain=0.5)
        nn.init.xavier_uniform_(self.control_lift.weight, gain=0.5)
        nn.init.xavier_uniform_(self.decoder.weight, gain=0.2)
        nn.init.zeros_(self.decoder.bias)

    def initial_hidden(self, batch_size: int, device: torch.device):
        return None

    def _state_features(self, obs_norm: torch.Tensor) -> torch.Tensor:
        pairwise: list[torch.Tensor] = []
        for i in range(self.obs_dim):
            for j in range(i + 1, self.obs_dim):
                pairwise.append(obs_norm[:, i : i + 1] * obs_norm[:, j : j + 1])
        pieces = [
            torch.ones(obs_norm.shape[0], 1, dtype=obs_norm.dtype, device=obs_norm.device),
            obs_norm,
            obs_norm * obs_norm,
            torch.sin(obs_norm),
            torch.cos(obs_norm) - 1.0,
        ]
        if pairwise:
            pieces.append(torch.cat(pairwise, dim=-1))
        return torch.cat(pieces, dim=-1)

    def _control_features(self, obs_norm: torch.Tensor, act_norm: torch.Tensor) -> torch.Tensor:
        return torch.cat([act_norm, act_norm * act_norm, obs_norm * act_norm], dim=-1)

    def forward(self, obs_norm: torch.Tensor, act_norm: torch.Tensor, hidden=None):
        z = self.state_lift(self._state_features(obs_norm))
        z_next = self.transition(z) + self.control_lift(self._control_features(obs_norm, act_norm))
        raw_delta = self.decoder(z_next)
        delta = self.delta_limit * torch.tanh(raw_delta / self.delta_limit)
        return delta, hidden
