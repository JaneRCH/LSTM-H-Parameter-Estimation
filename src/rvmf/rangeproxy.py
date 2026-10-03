r"""Daily volatility proxies from OHLC bars.

Three proxies are used, all as logs of a daily variance estimate.

``rv``
    The squared daily log return, :math:`r_t^2`.  This is the estimand of the
    interim report and of :cite:`MpandaGorjao2025`.  At constant volatility
    :math:`\log r_t^2 = \log \sigma_t^2 + \log z_t^2` with
    :math:`\operatorname{Var}(\log z_t^2) = \pi^2/2`, so it is the noisiest of
    the three by a wide margin.

``parkinson``
    :math:`(\log(H_t/L_t))^2 / (4\log 2)` :cite:`Parkinson1980`.

``garman_klass``
    :math:`\tfrac12 (\log(H_t/L_t))^2 - (2\log 2 - 1)(\log(C_t/O_t))^2`
    :cite:`GarmanKlass1980`.  This is the primary proxy, following
    :cite:`WuMuzyBacry2022`, who recover index roughness from Yahoo Finance
    daily OHLC with it.

The range proxies use the whole intraday path rather than two points of it, so
their sampling noise is far smaller.  That difference is what makes the latent
parameters identifiable at all from daily data, and it is quantified in the
report rather than assumed.
"""

from __future__ import annotations

import numpy as np

__all__ = ["PROXIES", "log_proxy", "proxy_variance"]

_GK_C = 2.0 * np.log(2.0) - 1.0
_FLOOR = 1e-12

PROXIES = ("rv", "parkinson", "garman_klass")


def proxy_variance(
    bars: dict[str, np.ndarray] | "object", kind: str
) -> np.ndarray:
    """Daily variance estimate of the named kind, on the natural scale."""
    o, h, l, c = (_column(bars, k) for k in ("open", "high", "low", "close"))
    hl = h - l
    co = c - o
    if kind == "rv":
        return co ** 2
    if kind == "parkinson":
        return hl ** 2 / (4.0 * np.log(2.0))
    if kind == "garman_klass":
        return 0.5 * hl ** 2 - _GK_C * co ** 2
    raise ValueError(f"unknown proxy {kind!r}; expected one of {PROXIES}")


def log_proxy(bars: dict[str, np.ndarray] | "object", kind: str) -> np.ndarray:
    """Log daily variance estimate, floored so that zeros do not propagate.

    Garman-Klass can return a small negative value when the close-to-open move
    nearly spans the whole range; those observations are floored rather than
    dropped, so the series keeps its calendar alignment.
    """
    v = proxy_variance(bars, kind)
    return np.log(np.maximum(v, _FLOOR))


def _column(bars, key: str) -> np.ndarray:
    if isinstance(bars, dict):
        return np.asarray(bars[key], dtype=float)
    return np.asarray(getattr(bars, key) if hasattr(bars, key) else bars[key], dtype=float)
