r"""Inputs to the learned estimator: a coarse-grained sequence and scaling summaries.

Two branches, following :cite:`Boros2024` for the invariance design and
:cite:`WuMuzyBacry2022` for the choice of summaries.

Sequence branch
    The three log proxies, coarse-grained into blocks of ``block`` days.  Each
    block contributes its mean and its within-block standard deviation, so the
    coarse-graining keeps the dispersion that a plain subsample would discard.
    Coarse-graining is also the natural operation for a scaling problem: it is
    how the multifractal formalism is defined.  A 5,100-day series becomes 510
    time steps of six channels, which an LSTM can traverse at a practical cost.

Summary branch
    The structure functions :math:`\log m(q,\Delta) = \log E|y_{t+\Delta}-y_t|^q`
    and the autocovariance differences :math:`D(n) = \hat C(n) - \hat C(0)` of
    the primary log proxy.  :cite:`WuMuzyBacry2022` give the expected form of
    the first: :math:`H` sets its slope, :math:`\lambda^2` its level and
    curvature, and measurement noise adds a constant at every lag.  The curve
    therefore contains the information the task needs, and its noise offset is
    also why a naive regression on it is biased.

Both branches are invariant to the volatility level by construction.  The
structure functions and the autocovariance differences are built from
differences of the log proxy, which removes the unknown mean, and the sequence
branch is standardised per sequence inside the network.  The level is a
nuisance parameter and must not reach the output.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .rangeproxy import PROXIES, log_proxy

__all__ = ["FeatureSpecV2", "sequence_features", "summary_features", "extract"]


def _log_grid(stop: int, count: int) -> np.ndarray:
    """Deduplicated logarithmic integer grid from 1 to ``stop``."""
    raw = np.unique(np.round(np.geomspace(1, stop, count)).astype(int))
    return raw[raw >= 1]


@dataclass(frozen=True)
class FeatureSpecV2:
    """Configuration of both input branches."""

    block: int = 10
    channels: tuple[str, ...] = PROXIES
    primary: str = "garman_klass"
    orders: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 3.0)
    max_delta: int = 100
    n_delta: int = 12
    max_lag: int = 100
    n_lag: int = 12
    deltas: np.ndarray = field(init=False, repr=False)
    lags: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "deltas", _log_grid(self.max_delta, self.n_delta))
        object.__setattr__(self, "lags", _log_grid(self.max_lag, self.n_lag))

    @property
    def n_sequence_features(self) -> int:
        return 2 * len(self.channels)

    @property
    def n_summary_features(self) -> int:
        return len(self.orders) * self.deltas.size + self.lags.size


def sequence_features(bars, spec: FeatureSpecV2) -> np.ndarray:
    """Coarse-grained sequence of shape ``(n_blocks, 2 * n_channels)``."""
    cols = []
    for name in spec.channels:
        y = log_proxy(bars, name)
        n_blocks = y.size // spec.block
        grid = y[: n_blocks * spec.block].reshape(n_blocks, spec.block)
        cols.append(grid.mean(axis=1))
        cols.append(grid.std(axis=1))
    return np.stack(cols, axis=1).astype(np.float32)


def summary_features(bars, spec: FeatureSpecV2) -> np.ndarray:
    """Structure functions and autocovariance differences of the primary proxy."""
    y = np.asarray(log_proxy(bars, spec.primary), dtype=float)

    out = []
    for q in spec.orders:
        for d in spec.deltas:
            diff = np.abs(y[int(d) :] - y[: y.size - int(d)])
            out.append(np.log(max(float(np.mean(diff ** q)), 1e-12)))

    centred = y - y.mean()
    c0 = float(centred @ centred) / centred.size
    for k in spec.lags:
        k = int(k)
        ck = float(centred[: centred.size - k] @ centred[k:]) / centred.size
        out.append(ck - c0)

    return np.asarray(out, dtype=np.float32)


def extract(bars, spec: FeatureSpecV2) -> tuple[np.ndarray, np.ndarray]:
    """Return both branches for one simulated or observed set of bars."""
    return sequence_features(bars, spec), summary_features(bars, spec)
