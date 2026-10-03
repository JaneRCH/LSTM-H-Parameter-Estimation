"""Vector-output LSTM estimator of the generalised Hurst function.

The network maps a sequence of multi-scale feature vectors to the whole vector
``{h(q)}`` on the moment grid, rather than to a single scalar.  Every published
learned estimator of the Hurst exponent targets the scalar; the vector is the
estimation target this study adds.

The anchored pair ``(H, lambda)`` is not a separate output head.  It is derived
from the predicted vector by :func:`rvmf.ghe.anchored_fit`, so the vector
benchmark against MF-DFA is preserved intact and the two-parameter summary comes
for free from the same forward pass.

Sequence reduction is by mean pooling over hidden states rather than by taking
the last state, because ``h(q)`` is a property of the whole path and not of its
endpoint; the last-state variant is retained for the ablation.

Dropout is not used.  It degraded performance in the Hurst-estimation setting of
Boros et al. (2024), and generalisation is instead protected by simulating every
training batch afresh, so there is no fixed training set to overfit.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import torch
from torch import nn

__all__ = ["ModelConfig", "HurstLSTM", "masked_mse", "predict_numpy"]


@dataclass
class ModelConfig:
    """Architecture and optimisation settings."""

    n_features: int = 44
    n_outputs: int = 11
    hidden: int = 128
    layers: int = 2
    pooling: str = "mean"  # "mean" | "last"
    lr: float = 3e-4
    batch_size: int = 64
    steps: int = 2000
    warmup: int = 50
    seed: int = 4833
    grad_clip: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


class HurstLSTM(nn.Module):
    """Two-layer LSTM with a dense head producing the ``h(q)`` vector."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.lstm = nn.LSTM(
            input_size=cfg.n_features,
            hidden_size=cfg.hidden,
            num_layers=cfg.layers,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(cfg.hidden, cfg.hidden // 2),
            nn.GELU(),
            nn.Linear(cfg.hidden // 2, cfg.n_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(batch, seq, features)`` to ``(batch, n_outputs)``."""
        out, _ = self.lstm(x)
        if self.cfg.pooling == "mean":
            pooled = out.mean(dim=1)
        elif self.cfg.pooling == "last":
            pooled = out[:, -1, :]
        else:
            raise ValueError(f"unknown pooling {self.cfg.pooling!r}")
        return self.head(pooled)


def masked_mse(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Mean squared error over the entries where the label is valid.

    The lognormal family has no valid label outside the moment range where its
    cascade has finite moments, so those entries are masked rather than fitted
    to an extrapolation.  Orders are otherwise unweighted, so the extreme
    moments are not down-weighted relative to the centre.
    """
    err = (pred - target) ** 2
    m = mask.to(err.dtype)
    denom = m.sum().clamp_min(1.0)
    return (err * m).sum() / denom


@torch.no_grad()
def predict_numpy(
    model: HurstLSTM, features: np.ndarray, batch_size: int = 64
) -> np.ndarray:
    """Predict ``h(q)`` for a stack of feature sequences."""
    model.eval()
    out = []
    for i in range(0, len(features), batch_size):
        chunk = torch.from_numpy(features[i : i + batch_size]).float()
        out.append(model(chunk).cpu().numpy())
    return np.concatenate(out, axis=0)
