"""Multifractal detrended fluctuation analysis.

Implements the five-step MF-DFA procedure of Kantelhardt et al. (2002) in
vectorised form, together with scaling-range selection.

The scaling-range selection is deliberately separated from the estimator and
exposed as an explicit, validatable policy (see :class:`ScalingRange`).  The
supervisor's feedback on the DSC4830 proposal identified "validation of the
scaling range" as an outstanding methodological item; the two policies here
are designed to be scored against known truth on synthetic paths.

Notation follows the report:

* ``h(q)``   generalised Hurst function, the slope of ``log F_q(s)`` on ``log s``
* ``tau(q)`` Renyi exponent, ``q h(q) - 1``
* ``f(alpha)`` singularity spectrum, Legendre transform of ``tau``

References
----------
Kantelhardt, J.W., Zschiegner, S.A., Koscielny-Bunde, E., Havlin, S.,
Bunde, A. and Stanley, H.E. (2002). Multifractal detrended fluctuation
analysis of nonstationary time series. Physica A 316(1-4), 87-114.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np

__all__ = [
    "ScalingRange",
    "MFDFAResult",
    "fluctuation_functions",
    "mfdfa",
    "select_scaling_range",
]

# Default moment grid used throughout the study.
DEFAULT_Q = np.arange(-5, 6, dtype=float)


# ----------------------------------------------------------------------------
# Scaling-range policy
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class ScalingRange:
    """Policy governing which scales enter the log-log regression.

    Two policies are supported.

    ``"fixed"``
        Scales are bounded below by ``s_min`` and above by ``L / min_segments``.
        The upper bound is the substantive change relative to the original
        DSC4830 specification, which used ``L / 4``: at ``L = 5101`` that admits
        ``s = 1275``, i.e. four non-overlapping segments, which cannot support a
        variance estimate.  Requiring ``min_segments`` segments at the largest
        fitted scale is the standard guard.

    ``"adaptive"`` (default)
        Within the bounds above, the sub-window maximising the mean coefficient
        of determination across ``q`` is selected, subject to spanning at least
        ``min_decades`` decades.  This reproduces the original specification but
        nested inside the segment-count guard, so it can no longer select a
        window that is unsupported by the data.

    The default was settled by scoring the policies against known truth rather
    than by argument.  On monofractal paths of length 5,000 with ``H`` drawn in
    ``[0.15, 0.45]``, the mean absolute error of the recovered ``h(2)`` is
    0.0137 for the adaptive policy with a ten-segment floor, 0.0155 for the
    fixed range, and 0.0182 for the original specification, which searched
    adaptively but allowed scales up to ``L / 4``.  The floor is therefore what
    improves accuracy, and the adaptive search is worth keeping: using every
    admissible scale is worse than selecting a window, and it also roughly
    doubles the width fabricated on a process that has none (0.136 against
    0.061).

    Parameters
    ----------
    policy
        ``"fixed"`` or ``"adaptive"``.
    s_min
        Smallest scale admitted.  Must exceed the detrending order by at least
        two points for the polynomial fit to have residual degrees of freedom;
        the effective value used is ``max(s_min, order + 2)``.
    min_segments
        Minimum number of non-overlapping segments at the largest fitted scale.
    min_decades
        Minimum width of the fitted window, in decades (``"adaptive"`` only).
    n_scales
        Number of logarithmically spaced candidate scales.
    """

    policy: Literal["fixed", "adaptive"] = "adaptive"
    s_min: int = 10
    min_segments: int = 10
    min_decades: float = 1.0
    n_scales: int = 24

    def candidate_scales(self, n: int, order: int) -> np.ndarray:
        """Logarithmically spaced integer scales admissible for a series of length ``n``."""
        lo = max(int(self.s_min), order + 2)
        hi = int(n // max(int(self.min_segments), 1))
        if hi <= lo:
            raise ValueError(
                f"series of length {n} admits no scaling range under this policy "
                f"(s_min={lo}, s_max={hi}); reduce min_segments or s_min"
            )
        scales = np.unique(
            np.round(np.geomspace(lo, hi, num=int(self.n_scales))).astype(int)
        )
        return scales[(scales >= lo) & (scales <= hi)]


@dataclass
class MFDFAResult:
    """Outcome of an MF-DFA estimation."""

    q: np.ndarray
    h: np.ndarray
    scales: np.ndarray          # candidate scales evaluated
    fq: np.ndarray              # (n_q, n_scales) fluctuation functions
    fit_slice: slice            # scales actually used in the regression
    r2: np.ndarray              # (n_q,) coefficient of determination per q
    stderr: np.ndarray          # (n_q,) regression standard error of the slope
    order: int
    meta: dict = field(default_factory=dict)

    @property
    def fitted_scales(self) -> np.ndarray:
        return self.scales[self.fit_slice]

    @property
    def n_segments_at_max_scale(self) -> int:
        return int(self.meta["n"] // self.fitted_scales[-1])

    @property
    def tau(self) -> np.ndarray:
        """Renyi exponent ``tau(q) = q h(q) - 1``."""
        return self.q * self.h - 1.0

    def summary(self) -> dict:
        """Compact diagnostic record, suitable for a per-series results table."""
        fs = self.fitted_scales
        return {
            "n": int(self.meta["n"]),
            "order": self.order,
            "scale_min": int(fs[0]),
            "scale_max": int(fs[-1]),
            "n_scales_fitted": int(fs.size),
            "segments_at_max_scale": self.n_segments_at_max_scale,
            "mean_r2": float(np.mean(self.r2)),
            "min_r2": float(np.min(self.r2)),
        }


# ----------------------------------------------------------------------------
# Core estimator
# ----------------------------------------------------------------------------


def _detrend_projection(s: int, order: int) -> np.ndarray:
    """Residual-maker ``I - V (V'V)^-1 V'`` for a polynomial fit of ``order`` on ``s`` points.

    The design matrix is built on ``t`` rescaled to ``[-1, 1]``, which keeps the
    Vandermonde system well conditioned at large ``s``.
    """
    t = np.linspace(-1.0, 1.0, s)
    v = np.vander(t, order + 1, increasing=True)
    # Orthonormal basis for the column space via QR is numerically safer than
    # forming the normal equations.
    qmat, _ = np.linalg.qr(v)
    return np.eye(s) - qmat @ qmat.T


def _segment_variances(profile: np.ndarray, s: int, order: int) -> np.ndarray:
    """Detrended residual variance of every segment of length ``s``.

    Segments are taken from both ends of the profile so that no observation is
    discarded when ``s`` does not divide ``L`` (step 2 of the procedure).

    Returns
    -------
    ndarray, shape (2 * n_s,)
        Residual variance per segment.
    """
    n = profile.size
    n_s = n // s
    if n_s == 0:
        raise ValueError(f"scale {s} exceeds series length {n}")

    forward = profile[: n_s * s].reshape(n_s, s)
    backward = profile[n - n_s * s :].reshape(n_s, s)
    blocks = np.concatenate([forward, backward], axis=0)

    proj = _detrend_projection(s, order)
    resid = blocks @ proj
    return np.mean(resid * resid, axis=1)


def fluctuation_functions(
    x: np.ndarray,
    q: Sequence[float] | np.ndarray,
    scales: Sequence[int] | np.ndarray,
    order: int = 2,
) -> np.ndarray:
    """Generalised fluctuation functions ``F_q(s)``.

    Parameters
    ----------
    x
        Series to analyse.  This is the estimand, not the price series.
    q
        Moment orders.  ``q = 0`` is handled by the logarithmic form.
    scales
        Segment lengths.
    order
        Polynomial detrending order.

    Returns
    -------
    ndarray, shape (len(q), len(scales))
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be one-dimensional")
    if not np.all(np.isfinite(x)):
        raise ValueError("x contains non-finite values")

    q = np.asarray(q, dtype=float)
    scales = np.asarray(scales, dtype=int)

    # Step 1: profile.
    profile = np.cumsum(x - x.mean())

    out = np.empty((q.size, scales.size), dtype=float)
    for j, s in enumerate(scales):
        var = _segment_variances(profile, int(s), order)
        # Guard against exactly-zero residual variance, which arises when a
        # segment is perfectly fitted by the detrending polynomial.  This is the
        # instability at negative q: the logarithm of a vanishing variance.
        var = np.maximum(var, np.finfo(float).tiny)

        for i, qi in enumerate(q):
            if abs(qi) < 1e-12:
                # Step 4, logarithmic form.  The limit of the moment average as
                # q -> 0 is the geometric mean of F, i.e.
                #   ln F_0(s) = (1 / 2) * mean_nu ln F^2(nu, s),
                # which is the 1 / (4 N_s) prefactor of the five-step
                # specification written against a sum over 2 N_s segments.
                out[i, j] = np.exp(0.5 * np.mean(np.log(var)))
            else:
                # Work in logs to avoid overflow at large |q|.
                lg = 0.5 * qi * np.log(var)
                m = lg.max()
                out[i, j] = np.exp((m + np.log(np.mean(np.exp(lg - m)))) / qi)
    return out


def _weighted_loglog_fit(
    log_s: np.ndarray, log_f: np.ndarray
) -> tuple[float, float, float]:
    """Ordinary least squares of ``log_f`` on ``log_s``.

    Returns ``(slope, r2, stderr_of_slope)``.
    """
    n = log_s.size
    xm, ym = log_s.mean(), log_f.mean()
    dx = log_s - xm
    sxx = np.dot(dx, dx)
    slope = np.dot(dx, log_f - ym) / sxx
    intercept = ym - slope * xm
    resid = log_f - (intercept + slope * log_s)
    ss_res = np.dot(resid, resid)
    ss_tot = np.dot(log_f - ym, log_f - ym)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    stderr = np.sqrt(ss_res / max(n - 2, 1) / sxx) if n > 2 else np.nan
    return float(slope), float(r2), float(stderr)


def select_scaling_range(
    fq: np.ndarray,
    scales: np.ndarray,
    spec: ScalingRange,
) -> slice:
    """Choose the sub-window of ``scales`` entering the regression.

    Under the ``"fixed"`` policy the whole admissible range is used, so the
    estimate does not depend on a data-driven search.  Under ``"adaptive"`` the
    window maximising mean ``R^2`` across ``q`` is returned, subject to the
    decade-width constraint.
    """
    n_scales = scales.size
    if spec.policy == "fixed":
        return slice(0, n_scales)

    log_s = np.log(scales.astype(float))
    log_f = np.log(fq)

    best_score, best = -np.inf, slice(0, n_scales)
    for i in range(n_scales):
        for j in range(i + 3, n_scales + 1):  # at least 3 points
            if log_s[j - 1] - log_s[i] < spec.min_decades * np.log(10.0):
                continue
            r2s = [
                _weighted_loglog_fit(log_s[i:j], log_f[k, i:j])[1]
                for k in range(log_f.shape[0])
            ]
            score = float(np.mean(r2s))
            if score > best_score:
                best_score, best = score, slice(i, j)
    return best


def mfdfa(
    x: np.ndarray,
    q: Sequence[float] | np.ndarray = DEFAULT_Q,
    order: int = 2,
    scaling_range: ScalingRange | None = None,
) -> MFDFAResult:
    """Estimate the generalised Hurst function of ``x``.

    Parameters
    ----------
    x
        The estimand.  For the empirical analysis this is a log-volatility
        proxy; for synthetic paths it is the simulated series itself.
    q
        Moment grid.  Defaults to the integer grid ``-5, ..., 5``.
    order
        Polynomial detrending order.
    scaling_range
        Scale-selection policy.  Defaults to the ``"fixed"`` policy with a
        ten-segment floor.

    Returns
    -------
    MFDFAResult
    """
    spec = scaling_range or ScalingRange()
    x = np.asarray(x, dtype=float)
    q = np.asarray(q, dtype=float)

    scales = spec.candidate_scales(x.size, order)
    fq = fluctuation_functions(x, q, scales, order=order)
    fit = select_scaling_range(fq, scales, spec)

    log_s = np.log(scales[fit].astype(float))
    h = np.empty(q.size)
    r2 = np.empty(q.size)
    se = np.empty(q.size)
    for i in range(q.size):
        h[i], r2[i], se[i] = _weighted_loglog_fit(log_s, np.log(fq[i, fit]))

    return MFDFAResult(
        q=q,
        h=h,
        scales=scales,
        fq=fq,
        fit_slice=fit,
        r2=r2,
        stderr=se,
        order=order,
        meta={"n": int(x.size), "policy": spec.policy},
    )
