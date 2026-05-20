"""Physics-inspired residual world model.

This branch uses a structured cart-pole update: kinematic position/angle
increments plus learned acceleration and residual polynomial terms. It avoids
the generic deep MLP while keeping the required homework interface.
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

        self.pos_from_vel = nn.Parameter(torch.tensor(0.05))
        self.angle_from_angvel = nn.Parameter(torch.tensor(0.05))
        self.accel_head = nn.Linear(8, 2)
        self.residual_head = nn.Linear(16, obs_dim, bias=False)
        self.residual_scale = nn.Parameter(torch.full((obs_dim,), 0.05))

        nn.init.zeros_(self.accel_head.weight)
        nn.init.zeros_(self.accel_head.bias)
        nn.init.zeros_(self.residual_head.weight)

    def initial_hidden(self, batch_size: int, device: torch.device):
        return None

    def _residual_features(self, obs_norm: torch.Tensor, act_norm: torch.Tensor) -> torch.Tensor:
        x = obs_norm[:, 0:1]
        theta = obs_norm[:, 1:2]
        x_dot = obs_norm[:, 2:3]
        theta_dot = obs_norm[:, 3:4]
        return torch.cat(
            [
                obs_norm,
                act_norm,
                theta * theta,
                x_dot * x_dot,
                theta_dot * theta_dot,
                act_norm * act_norm,
                theta * act_norm,
                x_dot * act_norm,
                theta_dot * act_norm,
                theta * theta_dot,
                x * x_dot,
                torch.sin(theta),
                torch.cos(theta) - 1.0,
            ],
            dim=-1,
        )

    def forward(self, obs_norm: torch.Tensor, act_norm: torch.Tensor, hidden=None):
        x = obs_norm[:, 0:1]
        theta = obs_norm[:, 1:2]
        x_dot = obs_norm[:, 2:3]
        theta_dot = obs_norm[:, 3:4]
        accel_features = torch.cat(
            [
                theta,
                x_dot,
                theta_dot,
                act_norm,
                theta * theta_dot,
                theta * act_norm,
                theta_dot * act_norm,
                torch.sin(theta),
            ],
            dim=-1,
        )
        accel = self.accel_head(accel_features)
        base_delta = torch.cat(
            [
                self.pos_from_vel * x_dot,
                self.angle_from_angvel * theta_dot,
                accel,
            ],
            dim=-1,
        )
        residual = self.residual_head(self._residual_features(obs_norm, act_norm))
        raw_delta = base_delta + self.residual_scale * residual
        delta = self.delta_limit * torch.tanh(raw_delta / self.delta_limit)
        return delta, hidden
