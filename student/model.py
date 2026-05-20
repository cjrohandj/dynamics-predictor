"""LSTM world model.

The homework interface only permits a tensor hidden state, so this branch packs
the LSTM hidden and cell states into one tensor of shape [batch, 2 * hidden].
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
        self.use_gru = True
        self.hidden_dim = int(hidden_dim)
        self.delta_limit = float(delta_limit)
        in_dim = obs_dim + act_dim
        self.input_proj = nn.Linear(in_dim, self.hidden_dim)
        self.cell = nn.LSTMCell(self.hidden_dim, self.hidden_dim)
        self.skip = nn.Linear(in_dim, obs_dim, bias=False)
        self.head = nn.Linear(self.hidden_dim, obs_dim)

    def initial_hidden(self, batch_size: int, device: torch.device):
        return torch.zeros(batch_size, 2 * self.hidden_dim, device=device)

    def forward(self, obs_norm: torch.Tensor, act_norm: torch.Tensor, hidden=None):
        x = torch.cat([obs_norm, act_norm], dim=-1)
        if hidden is None or hidden.numel() == 0:
            hidden = self.initial_hidden(obs_norm.shape[0], obs_norm.device)
        h, c = torch.chunk(hidden, 2, dim=-1)
        inp = torch.tanh(self.input_proj(x))
        h, c = self.cell(inp, (h, c))
        next_hidden = torch.cat([h, c], dim=-1)
        raw_delta = self.skip(x) + self.head(h)
        delta = self.delta_limit * torch.tanh(raw_delta / self.delta_limit)
        return delta, next_hidden
