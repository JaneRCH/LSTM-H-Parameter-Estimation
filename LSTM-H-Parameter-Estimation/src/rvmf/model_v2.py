r"""Two-branch estimator of the latent parameters ``(H, log lambda^2)``.

The sequence branch reads the coarse-grained log proxies with an LSTM; the
summary branch reads the structure functions and autocovariance differences
with a small multilayer perceptron.  The two representations are concatenated
and mapped to the two targets.

Invariance is built into the network rather than into preprocessing, following
:cite:`Boros2024`.  The sequence is standardised per channel per example inside
the forward pass, so neither the volatility level nor its scale can reach the
output.  Both are nuisance parameters here: the mean log variance is a function
of the unknown level and of the equally unknown correlation scale, and letting
either leak in would let the network learn the prior rather than the data.

The targets are standardised with fixed constants derived from the prior, not
from the batch, so that an estimate made on a single empirical series uses the
same scaling as training.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

__all__ = ["ModelConfigV2", "ThetaNet", "standardise_targets", "destandardise_targets",
           "predict_theta"]

#: Prior mean and standard deviation of (H, log lambda^2), used to standardise
#: the regression targets. Computed once from 200,000 draws of PriorSpec on the
#: 'prior_predictive' seed stream and frozen here, so that training and
#: single-series inference share a scale that does not depend on the batch.
TARGET_MEAN = np.array([0.1989, -3.4565], dtype=np.float32)
TARGET_STD = np.array([0.1566, 1.0645], dtype=np.float32)


@dataclass(frozen=True)
class ModelConfigV2:
    """Network size and optimisation settings."""

    n_sequence_features: int
    n_summary_features: int
    hidden: int = 128
    layers: int = 2
    summary_hidden: int = 64
    head_hidden: int = 64
    dropout: float = 0.0
    lr: float = 1e-3
    batch_size: int = 64
    steps: int = 3000
    grad_clip: float = 1.0
    use_sequence: bool = True
    use_summary: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


class ThetaNet(nn.Module):
    """Maps one set of daily bars to ``(H, log lambda^2)``."""

    def __init__(self, cfg: ModelConfigV2) -> None:
        super().__init__()
        if not (cfg.use_sequence or cfg.use_summary):
            raise ValueError("at least one branch must be enabled")
        self.cfg = cfg
        width = 0

        if cfg.use_sequence:
            self.lstm = nn.LSTM(
                input_size=cfg.n_sequence_features,
                hidden_size=cfg.hidden,
                num_layers=cfg.layers,
                batch_first=True,
                dropout=cfg.dropout if cfg.layers > 1 else 0.0,
            )
            width += cfg.hidden

        if cfg.use_summary:
            self.summary_mlp = nn.Sequential(
                nn.Linear(cfg.n_summary_features, cfg.summary_hidden),
                nn.SiLU(),
                nn.Linear(cfg.summary_hidden, cfg.summary_hidden),
                nn.SiLU(),
            )
            width += cfg.summary_hidden

        self.head = nn.Sequential(
            nn.Linear(width, cfg.head_hidden),
            nn.SiLU(),
            nn.Linear(cfg.head_hidden, 2),
        )

    @staticmethod
    def _standardise(x: torch.Tensor) -> torch.Tensor:
        """Per-example, per-channel standardisation of a ``(B, T, C)`` batch."""
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True).clamp_min(1e-6)
        return (x - mean) / std

    def forward(self, sequence: torch.Tensor, summary: torch.Tensor) -> torch.Tensor:
        parts = []
        if self.cfg.use_sequence:
            out, _ = self.lstm(self._standardise(sequence))
            parts.append(out.mean(dim=1))
        if self.cfg.use_summary:
            parts.append(self.summary_mlp(summary))
        return self.head(torch.cat(parts, dim=1))


def standardise_targets(y: np.ndarray) -> np.ndarray:
    """Scale ``(H, log lambda^2)`` to roughly zero mean and unit variance."""
    return (np.asarray(y, dtype=np.float32) - TARGET_MEAN) / TARGET_STD


def destandardise_targets(z: np.ndarray) -> np.ndarray:
    """Invert :func:`standardise_targets`."""
    return np.asarray(z, dtype=np.float32) * TARGET_STD + TARGET_MEAN


@torch.no_grad()
def predict_theta(
    model: ThetaNet, sequence: np.ndarray, summary: np.ndarray, batch: int = 256
) -> np.ndarray:
    """Predict ``(H, lambda^2)`` for a stack of examples.

    ``H`` is clipped to the support of the prior, which the model has no reason
    to leave and where the underlying process is not defined.
    """
    model.eval()
    out = []
    for i in range(0, len(sequence), batch):
        s = torch.from_numpy(np.asarray(sequence[i : i + batch], dtype=np.float32))
        m = torch.from_numpy(np.asarray(summary[i : i + batch], dtype=np.float32))
        out.append(model(s, m).cpu().numpy())
    z = destandardise_targets(np.concatenate(out))
    return np.column_stack([np.clip(z[:, 0], 0.0, 0.499), np.exp(z[:, 1])])
