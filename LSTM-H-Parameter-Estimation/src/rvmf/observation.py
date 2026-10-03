r"""From latent volatility to daily OHLC bars, in the convention of the data.

The estimators in this study never see the latent log-volatility process.  They
see daily bars built from it.  Training a network on the latent path and then
applying it to bars is the misspecification that :cite:`Boros2024` show breaks
learned estimators, so the simulator reproduces the whole chain: latent spot
variance on an intraday grid, intraday log-price increments, then daily open,
high, low and close.

The bar convention follows the data rather than an idealisation.  In the four
JSE files the recorded open equals the previous close on 99.3 to 99.8 percent
of days, while the high equals ``max(open, close)`` on only 19 to 27 percent.
The open therefore carries no overnight information and the range does.  The
simulator sets :math:`O_t = C_{t-1}` and takes the high and low over the
previous close together with the day's intraday path, which is what produces
bars with those two properties.

A consequence for the interim report: the overnight term of the Yang-Zhang
estimator is identically zero on this data, so Yang-Zhang reduces to a
Rogers-Satchell variant and cannot be described as incorporating overnight
information.  Garman-Klass and Parkinson are the honest range proxies here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sfbm import SfbmParams, simulate_omega, spot_variance

__all__ = ["ObservationParams", "simulate_bars", "DEFAULT_STEPS_PER_DAY"]

DEFAULT_STEPS_PER_DAY = 48
"""Ten-minute steps over an eight-hour JSE session."""


@dataclass(frozen=True)
class ObservationParams:
    """Nuisance parameters of the price-formation and recording process.

    Attributes
    ----------
    rho
        Leverage.  Intraday shocks load on the contemporaneous change in log
        volatility with this correlation; negative for equities.
    kappa
        Opening-gap variance as a multiple of the day's spot variance.  The gap
        moves the price but is not recorded in the open, exactly as in the data.
    df
        Degrees of freedom of the Student-t innovations, rescaled to unit
        variance.  ``None`` gives Gaussian innovations.
    steps_per_day
        Intraday grid resolution.
    """

    rho: float = -0.4
    kappa: float = 0.1
    df: float | None = None
    steps_per_day: int = DEFAULT_STEPS_PER_DAY

    def __post_init__(self) -> None:
        if not (-1.0 < self.rho <= 0.0):
            raise ValueError(f"rho must lie in (-1, 0], got {self.rho}")
        if self.kappa < 0.0:
            raise ValueError(f"kappa must be non-negative, got {self.kappa}")
        if self.df is not None and self.df <= 2.0:
            raise ValueError(f"df must exceed 2 for finite variance, got {self.df}")
        if self.steps_per_day < 2:
            raise ValueError("steps_per_day must be at least 2")


def simulate_bars(
    n_days: int,
    sf: SfbmParams,
    obs: ObservationParams,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Simulate ``n_days`` daily bars and return them with the latent truth.

    Returns
    -------
    dict
        ``open``, ``high``, ``low``, ``close`` as log prices; ``ret`` as the
        daily log return ``close - open``; and ``integrated_variance``, the
        latent daily integrated variance, which is the quantity the proxies are
        noisy measurements of.
    """
    m = obs.steps_per_day
    total = n_days * m
    dt = 1.0 / m

    omega = simulate_omega(total, dt, sf, rng)
    var_spot = spot_variance(omega, sf)  # variance per day, mean sigma_bar^2/252

    # Leverage: load the innovation on the standardised change in log volatility.
    d_omega = np.diff(omega, prepend=omega[0])
    sd = d_omega.std()
    eps = d_omega / sd if sd > 0 else np.zeros_like(d_omega)
    xi = _unit_variance_innovations(total, obs.df, rng)
    z = obs.rho * eps + np.sqrt(1.0 - obs.rho ** 2) * xi

    increments = np.sqrt(var_spot * dt) * z
    grid = increments.reshape(n_days, m)

    # The opening gap moves the price but is never recorded in the open.
    daily_var = var_spot.reshape(n_days, m).mean(axis=1)
    gap = rng.standard_normal(n_days) * np.sqrt(obs.kappa * daily_var)

    within = np.cumsum(grid, axis=1) + gap[:, None]
    prev_close = np.concatenate([[0.0], np.cumsum(within[:, -1])[:-1]])

    path = prev_close[:, None] + within
    open_ = prev_close                                   # recorded open
    close = path[:, -1]
    high = np.maximum(path.max(axis=1), prev_close)      # range spans the stale open
    low = np.minimum(path.min(axis=1), prev_close)

    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "ret": close - open_,
        "integrated_variance": (var_spot * dt).reshape(n_days, m).sum(axis=1),
    }


def _unit_variance_innovations(
    n: int, df: float | None, rng: np.random.Generator
) -> np.ndarray:
    """Standard normal, or Student-t rescaled to unit variance."""
    if df is None:
        return rng.standard_normal(n)
    return rng.standard_t(df, size=n) / np.sqrt(df / (df - 2.0))
