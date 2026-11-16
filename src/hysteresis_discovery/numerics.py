"""Differentiable time-series operations shared by the experiments."""

import math

import torch


def rmse(prediction, target):
    """Compute RMSE with a finite zero-residual gradient.

    The equivalent sqrt(mean(error**2)) has an undefined intermediate
    derivative at zero, which can produce NaNs during exact initialization.
    """
    return torch.linalg.vector_norm(prediction - target) / math.sqrt(target.numel())


def lerp_series(series, t, t0, dt):
    """Linearly interpolate a uniformly sampled, one-dimensional tensor."""
    pos = (t - t0) / dt
    i0 = torch.clamp(pos.floor().long(), 0, series.numel() - 2)
    w = (pos - i0.to(pos.dtype)).clamp(0, 1)
    return (1 - w) * series[i0] + w * series[i0 + 1]


def diff_series(series, dt):
    """Use central differences internally and one-sided endpoint differences."""
    derivative = torch.empty_like(series)
    derivative[1:-1] = (series[2:] - series[:-2]) / (2 * dt)
    derivative[0] = (series[1] - series[0]) / dt
    derivative[-1] = (series[-1] - series[-2]) / dt
    return derivative
