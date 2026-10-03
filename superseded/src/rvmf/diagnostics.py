"""Distributional and stationarity diagnostics for the estimand."""

from __future__ import annotations

import numpy as np

__all__ = ["hill_estimator", "hill_plateau", "descriptives", "stationarity"]


def hill_estimator(x: np.ndarray, k: int | None = None, fraction: float = 0.05):
    """Hill estimate of the tail index of ``|x|``.

    Parameters
    ----------
    x
        Series whose tail index is wanted. The absolute values are used, so the
        estimate refers to the heavier of the two tails combined.
    k
        Number of upper order statistics. Overrides ``fraction``.
    fraction
        Share of the sample used, when ``k`` is not given.

    Returns
    -------
    float
        The estimated tail index ``gamma``, the reciprocal of the Hill
        statistic. Larger values mean lighter tails; ``gamma > 2`` implies a
        finite variance.
    """
    a = np.sort(np.abs(np.asarray(x, dtype=float)))
    a = a[a > 0]
    n = a.size
    if n < 20:
        return float("nan")
    kk = int(k if k is not None else max(10, int(fraction * n)))
    kk = min(kk, n - 1)
    top = a[-kk - 1 :]
    return float(1.0 / np.mean(np.log(top[1:] / top[0])))


def hill_plateau(x: np.ndarray, fractions=(0.01, 0.03, 0.05, 0.10, 0.15)) -> dict:
    """Hill estimates across a range of tail fractions.

    A tail index read off a single fraction is not trustworthy; the estimate is
    reported against the range over which it is stable.
    """
    values = {float(f): hill_estimator(x, fraction=f) for f in fractions}
    finite = [v for v in values.values() if np.isfinite(v)]
    return {
        "by_fraction": values,
        "min": float(np.min(finite)) if finite else float("nan"),
        "max": float(np.max(finite)) if finite else float("nan"),
    }


def descriptives(x: np.ndarray) -> dict:
    """Moments and a normality test."""
    from scipy import stats

    a = np.asarray(x, dtype=float)
    jb, jb_p = stats.jarque_bera(a)
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "sd": float(a.std(ddof=1)),
        "skewness": float(stats.skew(a)),
        "excess_kurtosis": float(stats.kurtosis(a)),
        "min": float(a.min()),
        "max": float(a.max()),
        "jarque_bera": float(jb),
        "jarque_bera_p": float(jb_p),
    }


def stationarity(x: np.ndarray) -> dict:
    """ADF and KPSS.

    A joint rejection is the signature of a stationary long-memory process,
    whose slowly decaying autocorrelation KPSS reads as a shifting level. It is
    reported as such rather than treated as a contradiction.
    """
    from statsmodels.tsa.stattools import adfuller, kpss

    a = np.asarray(x, dtype=float)
    adf_stat, adf_p, *_ = adfuller(a, autolag="AIC")
    kpss_stat, kpss_p, _, kpss_crit = kpss(a, regression="c", nlags="auto")
    return {
        "adf_stat": float(adf_stat),
        "adf_p": float(adf_p),
        "adf_rejects_unit_root": bool(adf_p < 0.05),
        "kpss_stat": float(kpss_stat),
        "kpss_p": float(kpss_p),
        "kpss_crit_5pct": float(kpss_crit["5%"]),
        "kpss_rejects_stationarity": bool(kpss_stat > kpss_crit["5%"]),
    }
