"""Multi-scale feature construction for the learned estimator.

The network never sees the raw series.  It sees a sequence of multi-scale
summaries, because a scaling exponent is a statement about how fluctuation size
grows with the observation horizon, and these features make that growth
directly readable.

At each time ``t`` the feature vector holds

* the log rolling variance of the estimand over windows ``w`` in ``{5, 22, 63,
  126}``, four values; and
* the log structure functions

      s(q, w)_t = log( (1/w) * sum_{tau = t-w+1}^{t} |dX_tau| ** q ),

  over the same windows and moment orders ``q`` in ``{+-1, ..., +-5}``, forty
  values.

The negative orders matter and are not decoration.  They carry the
small-fluctuation information that the positive orders cannot see, so every
output ``h(q)`` has an input measured at the same moment order rather than
being extrapolated from the positive side.  They are computed with a flooring
constant on ``|dX|``.

Standardisation
---------------
The scaling information lives in the *differences* between log features at
different windows, since a difference of logs is a log ratio.  Standardising
each feature by its own statistics would apply a different affine map per
window and would destroy exactly that structure.

Standardising by one statistic common to the whole matrix preserves those
differences but fails for a different reason, which was measured rather than
assumed: the structure functions at ``q = -5`` vary by around 1.9 in the log
domain while those at moderate orders vary by far less, so a single global
scale leaves 32 of the 44 columns with a standard deviation near ``1e-6``,
numerically invisible to the network.

The default therefore standardises by a statistic common to each *moment
order*, the four windows of a given ``q`` sharing one centre and one scale.
Cross-window differences, which are what encode the exponent, are preserved
exactly within each order, while orders of differing magnitude are placed on a
comparable footing.  The global and per-feature variants are retained as
ablations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["FeatureSpec", "extract_features", "feature_names"]

DEFAULT_WINDOWS = (5, 22, 63, 126)
DEFAULT_ORDERS = (-5.0, -4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0, 5.0)


@dataclass(frozen=True)
class FeatureSpec:
    """Configuration of the multi-scale feature map."""

    windows: tuple[int, ...] = DEFAULT_WINDOWS
    orders: tuple[float, ...] = DEFAULT_ORDERS
    stride: int = 10
    floor: float = 1e-12
    standardise: str = "per_order"  # "per_order" | "common" | "per_feature" | "none"
    log_clip: float = 50.0

    @property
    def n_features(self) -> int:
        return len(self.windows) * (1 + len(self.orders))

    def sequence_length(self, n: int) -> int:
        start = max(self.windows)
        return max(0, (n - start) // self.stride)


def feature_names(spec: FeatureSpec) -> list[str]:
    """Human-readable names, in the column order produced by :func:`extract_features`.

    Columns are grouped by moment order, the log rolling variances forming their
    own group, so that order-wise standardisation is a contiguous slice.
    """
    names = [f"logvar_w{w}" for w in spec.windows]
    for q in spec.orders:
        names.extend(f"sf_q{q:+g}_w{w}" for w in spec.windows)
    return names


def feature_groups(spec: FeatureSpec) -> np.ndarray:
    """Group index per column: 0 for the log variances, then one per moment order."""
    groups = [0] * len(spec.windows)
    for i, _ in enumerate(spec.orders, start=1):
        groups.extend([i] * len(spec.windows))
    return np.asarray(groups, dtype=int)


def _rolling_mean(a: np.ndarray, w: int) -> np.ndarray:
    """Trailing rolling mean of width ``w``, valid positions only.

    Returns an array of length ``len(a) - w + 1`` where element ``i`` is the
    mean of ``a[i : i + w]``.  Uses a cumulative sum, so cost is linear in the
    series length and independent of the window.
    """
    c = np.cumsum(np.concatenate([[0.0], a]))
    return (c[w:] - c[:-w]) / float(w)


def extract_features(x: np.ndarray, spec: FeatureSpec | None = None) -> np.ndarray:
    """Build the multi-scale feature sequence for one series.

    Parameters
    ----------
    x
        The estimand, a one-dimensional series.
    spec
        Feature configuration.

    Returns
    -------
    ndarray, shape (sequence_length, n_features)
        Standardised features, ordered as :func:`feature_names`.
    """
    spec = spec or FeatureSpec()
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be one-dimensional")

    n = x.size
    max_w = max(spec.windows)
    if n <= max_w + spec.stride:
        raise ValueError(
            f"series of length {n} is too short for windows up to {max_w}"
        )

    dx = np.diff(x, prepend=x[0])
    abs_dx = np.maximum(np.abs(dx), spec.floor)
    log_abs = np.log(abs_dx)
    centred = x - x.mean()

    columns: list[np.ndarray] = []

    # Log rolling variance, computed as E[z^2] - (E[z])^2 on the centred series.
    for w in spec.windows:
        m1 = _rolling_mean(centred, w)
        m2 = _rolling_mean(centred * centred, w)
        var = np.maximum(m2 - m1 * m1, spec.floor)
        col = np.log(var)
        columns.append(np.concatenate([np.full(w - 1, col[0]), col]))

    # Log structure functions.  The exponent q * log|dx| is centred on its own
    # median before exponentiating, so the rolling mean is taken over values of
    # order one whatever q is.  Centring on a constant instead would push every
    # value below the floor at large |q| and pin the column to that floor, which
    # is a silent failure rather than a loud one.
    for q in spec.orders:
        scaled = q * log_abs
        centre = float(np.median(scaled))
        weights = np.exp(np.clip(scaled - centre, -spec.log_clip, spec.log_clip))
        for w in spec.windows:
            m = _rolling_mean(weights, w)
            col = np.log(np.maximum(m, spec.floor)) + centre
            columns.append(np.concatenate([np.full(w - 1, col[0]), col]))

    feats = np.column_stack(columns)

    # Drop the burn-in region, then stride.
    feats = feats[max_w:][:: spec.stride]

    if not np.all(np.isfinite(feats)):
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    if spec.standardise == "per_order":
        groups = feature_groups(spec)
        for g in np.unique(groups):
            block = feats[:, groups == g]
            sd = float(block.std())
            feats[:, groups == g] = (block - float(block.mean())) / (
                sd if sd > 0 else 1.0
            )
    elif spec.standardise == "common":
        mu = float(feats.mean())
        sd = float(feats.std())
        feats = (feats - mu) / (sd if sd > 0 else 1.0)
    elif spec.standardise == "per_feature":
        mu = feats.mean(axis=0, keepdims=True)
        sd = feats.std(axis=0, keepdims=True)
        feats = (feats - mu) / np.where(sd > 0, sd, 1.0)
    elif spec.standardise != "none":
        raise ValueError(f"unknown standardise mode {spec.standardise!r}")

    return feats.astype(np.float32)
