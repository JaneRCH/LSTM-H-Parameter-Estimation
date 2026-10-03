"""Volatility proxies and the estimand convention.

Two proxies are carried co-primary, as the design requires the central finding
to survive a change of measurement:

``rv``
    Realised volatility at the daily horizon, ``|r_t|``.  This is the
    construction used by the published JSE benchmark, so estimates are directly
    comparable with it.
``yz``
    The Yang-Zhang range estimator on a rolling window, which uses the full
    OHLC record and has the lowest estimation variance in its class.

Both are carried in logarithms by default.  The log transform is what makes the
estimate commensurable with the rough-volatility literature, which models
log-volatility rather than the level, and it is the transform used in the
supervisor's own SPX work.

The estimand convention
-----------------------
MF-DFA integrates its input before detrending, so the exponent it returns
depends on whether the input is a *series* or a *path*:

* fed a noise-like series (fGn), it returns ``H``;
* fed the corresponding path (fBm), it returns ``H + 1``.

This was verified numerically: at ``H`` of 0.2, 0.3 and 0.7 the two inputs
differ by 0.958, 0.984 and 0.973 respectively.  The convention is therefore not
cosmetic, and :class:`Estimand` makes it an explicit, recorded choice rather
than an implicit one.  Reporting ``h(2)`` without naming the convention is what
makes a level-series estimate near 0.6 look irreconcilable with a published
increment-convention band near 0.2 to 0.3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

__all__ = [
    "Estimand",
    "log_returns",
    "realised_volatility",
    "yang_zhang",
    "build_estimand",
]

ProxyName = Literal["rv", "yz"]
Convention = Literal["series", "increments"]


@dataclass(frozen=True)
class Estimand:
    """A fully specified estimand: proxy, transform and convention."""

    proxy: ProxyName = "rv"
    log: bool = True
    convention: Convention = "series"
    yz_window: int = 22
    floor_quantile: float = 1e-4

    def label(self) -> str:
        parts = [self.proxy, "log" if self.log else "level", self.convention]
        if self.proxy == "yz":
            parts.append(f"w{self.yz_window}")
        return "-".join(parts)


def log_returns(close: pd.Series) -> pd.Series:
    """Daily log-returns ``r_t = log(C_t / C_{t-1})``."""
    return np.log(close / close.shift(1)).dropna()


def realised_volatility(close: pd.Series) -> pd.Series:
    """Daily realised volatility ``|r_t|``."""
    return log_returns(close).abs().rename("rv")


def yang_zhang(ohlc: pd.DataFrame, window: int = 22) -> pd.Series:
    """Yang-Zhang volatility on a rolling window of ``window`` trading days.

    The estimator combines an overnight component, an open-to-close component
    and the Rogers-Satchell component, with the weight

        k = 0.34 / (1.34 + (n + 1) / (n - 1)).

    It is defined over a window rather than a single day, so the returned series
    carries ``window``-lag autocorrelation by construction.  Two consequences
    are handled downstream rather than ignored: the fitted scaling range is
    restricted to scales above ``window``, and Monte Carlo null replicates are
    passed through the identical window so the induced dependence enters null
    and observed statistic alike.

    Parameters
    ----------
    ohlc
        Frame with ``open``, ``high``, ``low``, ``close`` columns.
    window
        Rolling window length in trading days.

    Returns
    -------
    pandas.Series
        Volatility (not variance), indexed as ``ohlc`` with the first
        ``window`` observations dropped.
    """
    required = {"open", "high", "low", "close"}
    missing = required - set(ohlc.columns)
    if missing:
        raise ValueError(f"ohlc is missing columns: {sorted(missing)}")
    if window < 2:
        raise ValueError("window must be at least 2")

    o, h, l, c = (ohlc[k].astype(float) for k in ("open", "high", "low", "close"))
    prev_c = c.shift(1)

    overnight = np.log(o / prev_c)
    open_to_close = np.log(c / o)
    rogers_satchell = np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)

    n = window
    k = 0.34 / (1.34 + (n + 1) / (n - 1))

    var_o = overnight.rolling(n).var(ddof=1)
    var_c = open_to_close.rolling(n).var(ddof=1)
    var_rs = rogers_satchell.rolling(n).mean()

    var_yz = var_o + k * var_c + (1.0 - k) * var_rs
    # Rounding and stale quotes can push the combination marginally negative.
    var_yz = var_yz.where(var_yz > 0)
    return np.sqrt(var_yz).dropna().rename("yz")


def build_estimand(ohlc: pd.DataFrame, spec: Estimand) -> pd.Series:
    """Construct the series on which every exponent is estimated.

    Parameters
    ----------
    ohlc
        Frame with at least a ``close`` column; ``yz`` additionally requires
        ``open``, ``high`` and ``low``.
    spec
        The estimand specification.

    Returns
    -------
    pandas.Series
    """
    if spec.proxy == "rv":
        x = realised_volatility(ohlc["close"])
    elif spec.proxy == "yz":
        x = yang_zhang(ohlc, spec.yz_window)
    else:
        raise ValueError(f"unknown proxy {spec.proxy!r}")

    if spec.log:
        # Zero-return days make |r_t| exactly zero, so the logarithm needs a
        # floor.  A low quantile of the strictly positive values is used rather
        # than an arbitrary epsilon, so the floor scales with the series.
        positive = x[x > 0]
        if positive.empty:
            raise ValueError("estimand has no strictly positive values")
        floor = float(positive.quantile(spec.floor_quantile))
        x = np.log(x.clip(lower=floor))

    if spec.convention == "increments":
        x = x.diff().dropna()
    elif spec.convention != "series":
        raise ValueError(f"unknown convention {spec.convention!r}")

    return x.rename(spec.label())
