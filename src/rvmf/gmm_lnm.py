r"""GMM on the log integrated variance, after :cite:`WuMuzyBacry2022`.

Writing :math:`\bar\omega_t` for the average of the latent log volatility over
day :math:`t`, the observed log proxy is

.. math:: y_t = \bar\omega_t + c + \eta_t,

with :math:`\eta_t` an independent measurement error of variance :math:`V_1`.
The estimator fits the autocovariance differences
:math:`D(n) = \hat C(n) - \hat C(0)`, which are free of the unknown level
:math:`c`.

Averaging the S-fBM covariance over two unit windows gives, for
:math:`n \Delta \ll T`,

.. math::
    D(n; H, \lambda^2, V_1)
      = -\frac{\lambda^2}{2H(1-2H)}\bigl[f_H(n) - f_H(0)\bigr] - V_1,
    \qquad n \ge 1,

where :math:`f_H(n) = \varphi(n+1) - 2\varphi(n) + \varphi(n-1)` and
:math:`\varphi(x) = |x|^{2H+2} / [(2H+1)(2H+2)]`.  The triangular kernel is the
second-difference operator, so :math:`f_H` is the second difference of the
twice-integrated power.  In the multifractal limit the same construction with
:math:`\varphi(x) = x^2(2\log|x| - 3)/4` gives
:math:`D(n) = -\lambda^2[f_0(n) - f_0(0)] - V_1`.

Two properties matter for this study.  The correlation scale :math:`T` cancels
from :math:`D(n)` entirely, which is why :cite:`WuMuzyBacry2022` treat it as a
nuisance that cannot be estimated from a sample shorter than itself.  And the
measurement error enters as a constant shift, so modelling it costs one
parameter and removes the upward bias that defeats the naive scaling
regression.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

__all__ = ["GmmResult", "default_lags", "model_D", "estimate", "gjr_regression"]


def default_lags(max_lag: int = 512) -> np.ndarray:
    """The lag grid of :cite:`WuMuzyBacry2022`: ``floor(2**(k/2))``, deduplicated."""
    raw = np.floor(2.0 ** (np.arange(19) / 2.0)).astype(int)
    return np.unique(raw[raw <= max_lag])


def _phi(x: np.ndarray, H: float) -> np.ndarray:
    """Twice-integrated power, or its logarithmic limit at ``H = 0``."""
    ax = np.abs(x)
    if H <= 0.0:
        out = np.zeros_like(ax, dtype=float)
        nz = ax > 0.0
        out[nz] = ax[nz] ** 2 * (2.0 * np.log(ax[nz]) - 3.0) / 4.0
        return out
    return ax ** (2.0 * H + 2.0) / ((2.0 * H + 1.0) * (2.0 * H + 2.0))


def _f(n: np.ndarray, H: float) -> np.ndarray:
    """Unit-window average of the lag kernel, a second difference of ``phi``."""
    n = np.asarray(n, dtype=float)
    return _phi(n + 1.0, H) - 2.0 * _phi(n, H) + _phi(n - 1.0, H)


def model_D(lags: np.ndarray, H: float, lambda_sq: float, noise: float) -> np.ndarray:
    """Model autocovariance differences at the given lags."""
    lags = np.asarray(lags, dtype=float)
    amp = lambda_sq if H <= 0.0 else lambda_sq / (2.0 * H * (1.0 - 2.0 * H))
    return -amp * (_f(lags, H) - _f(np.array([0.0]), H)[0]) - noise


def empirical_D(y: np.ndarray, lags: np.ndarray) -> np.ndarray:
    """Sample autocovariance differences ``C(n) - C(0)`` of a series."""
    y = np.asarray(y, dtype=float)
    y = y - y.mean()
    n = y.size
    c0 = float(y @ y) / n
    return np.array([float(y[: n - int(k)] @ y[int(k) :]) / n - c0 for k in lags])


def _hac_weights(y: np.ndarray, lags: np.ndarray) -> np.ndarray:
    """Diagonal Newey-West weights, one per moment.

    The full HAC covariance of 19 highly collinear autocovariance moments is
    badly conditioned at these sample sizes, so the diagonal is used and the
    departure from the two-step GMM of :cite:`WuMuzyBacry2022` is recorded.
    Weighting by the inverse HAC variance of each moment keeps the long lags,
    which are the noisiest, from dominating the fit.
    """
    y = np.asarray(y, dtype=float)
    y = y - y.mean()
    n = y.size
    bandwidth = max(1, int(round(n ** (1.0 / 3.0))))
    var = np.empty(lags.size)
    for i, k in enumerate(lags):
        k = int(k)
        g = y[: n - k] * y[k:]
        g = g - g.mean()
        s = float(g @ g) / g.size
        for b in range(1, min(bandwidth, g.size - 1) + 1):
            w = 1.0 - b / (bandwidth + 1.0)
            s += 2.0 * w * float(g[: g.size - b] @ g[b:]) / g.size
        var[i] = max(s / g.size, 1e-12)
    return 1.0 / var


def gjr_regression(y: np.ndarray, lags: np.ndarray) -> float:
    """Naive scaling-regression estimate of ``H``, used only as a starting value.

    :cite:`WuMuzyBacry2022` show this is biased upwards by 0.03 to 0.08 for
    ``H`` below 0.15; it is never reported as an estimate here.
    """
    y = np.asarray(y, dtype=float)
    m = np.array([np.mean(np.abs(y[int(k) :] - y[: y.size - int(k)]) ** 2) for k in lags])
    good = m > 0
    if good.sum() < 3:
        return 0.1
    slope = np.polyfit(np.log(lags[good]), np.log(m[good]), 1)[0]
    return float(np.clip(slope / 2.0, 0.0, 0.49))


@dataclass(frozen=True)
class GmmResult:
    """Point estimates and fit diagnostics."""

    H: float
    lambda_sq: float
    noise: float
    objective: float
    success: bool


def estimate(
    y: np.ndarray, lags: np.ndarray | None = None, max_lag_frac: float = 0.1
) -> GmmResult:
    """Estimate ``(H, lambda^2, V1)`` from one log-proxy series.

    Parameters
    ----------
    y
        Log daily variance proxy.
    lags
        Lag grid; defaults to :func:`default_lags` truncated so that no lag
        exceeds ``max_lag_frac`` of the sample.
    """
    y = np.asarray(y, dtype=float)
    if lags is None:
        lags = default_lags(max_lag=max(2, int(max_lag_frac * y.size)))
    lags = np.asarray(lags, dtype=int)

    target = empirical_D(y, lags)
    weights = _hac_weights(y, lags)
    weights = weights / weights.sum()

    def objective(theta: np.ndarray) -> float:
        H, log_lam, noise = theta
        resid = model_D(lags, H, np.exp(log_lam), noise) - target
        return float(weights @ resid ** 2)

    start_H = gjr_regression(y, lags)
    best: GmmResult | None = None
    for H0 in (start_H, 0.02, 0.12, 0.30):
        for lam0 in (0.02, 0.08):
            res = minimize(
                objective,
                x0=np.array([H0, np.log(lam0), 0.3 * float(np.var(y))]),
                method="L-BFGS-B",
                bounds=[(0.0, 0.499), (np.log(1e-4), np.log(2.0)), (0.0, 5.0 * float(np.var(y)))],
            )
            cand = GmmResult(
                H=float(res.x[0]),
                lambda_sq=float(np.exp(res.x[1])),
                noise=float(res.x[2]),
                objective=float(res.fun),
                success=bool(res.success),
            )
            if best is None or cand.objective < best.objective:
                best = cand
    assert best is not None
    return best
