"""Acceleration networks for equation discovery."""

import torch
from torch import nn


class XDotDotNet(nn.Module):
    """Predict acceleration from displacement, velocity, hysteresis, and force."""

    def __init__(self, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, x_dot, z, u):
        inputs = torch.cat([value.unsqueeze(-1) for value in (x, x_dot, z, u)], dim=-1)
        return self.net(inputs).squeeze(-1)
