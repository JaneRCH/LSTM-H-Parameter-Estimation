"""Generalised Hurst exponent: spectrum, width, and the anchored (H, lambda) model.

Two complementary summaries of an estimated ``h(q)`` vector are provided.

The classical summary is the singularity spectrum ``f(alpha)`` obtained from the
Renyi exponent by Legendre transform, and its width ``Delta alpha``.

The second is the anchored linear model

    h(q) = H - lambda * qt + xi(q),      qt := q - 2,   H := h(2),

in which ``lambda`` is the effective multifractality strength, equal to the
negative local slope of ``h`` at ``q = 2``, and ``xi(q)`` collects departures
from linearity.  This is the parameterisation the supervisor asked for: it
replaces an eleven-component vector with two interpretable scalars plus a
residual, and it is the form the lognormal multifractal random walk satisfies
exactly (see :mod:`rvmf.simulators`).

.. warning::
   The weights ``c_q`` in the anchored estimator are implemented as
   ``c_q = -qt``, which is the ordinary least-squares solution of the anchored
   regression and is consistent with the ``sum(qt**2)`` denominator in the
   supervisor's equation (2.13).  The source document was available only as a
   screenshot, so this reading is not confirmed.  Pass ``weights`` explicitly if
   his definition differs.  The same caveat applies to the definition of the
   reported ``r2``; see :func:`anchored_fit`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = [
    "SpectrumResult",
    "AnchoredFit",
    "renyi_tau",
    "singularity_spectrum",
    "spectrum_width",
    "anchored_fit",
]


@dataclass
class SpectrumResult:
    """Singularity spectrum and its scalar summaries."""

    q: np.ndarray
    h: np.ndarray
    tau: np.ndarray
    alpha: np.ndarray
    f: np.ndarray

    @property
    def width(self) -> float:
        """``Delta alpha``, the spectrum width."""
        return float(np.max(self.alpha) - np.min(self.alpha))

    @property
    def asymmetry(self) -> float:
        """Spectrum asymmetry.

        Positive values indicate a longer left tail, i.e. dominance of the
        large-fluctuation side.  Reported alongside the width because the
        synthetic families used for training have symmetric spectra whereas
        empirical spectra generally do not.
        """
        a_max = float(self.alpha[np.argmax(self.f)])
        left = a_max - float(np.min(self.alpha))
        right = float(np.max(self.alpha)) - a_max
        denom = left + right
        return float((left - right) / denom) if denom > 0 else np.nan


@dataclass
class AnchoredFit:
    """Result of fitting ``h(q) = H - lambda * (q - 2) + xi(q)``."""

    H: float
    lam: float
    lam_plus: float
    xi: np.ndarray
    r2: float
    q: np.ndarray

    def predict(self) -> np.ndarray:
        return self.H - self.lam * (self.q - 2.0)


def renyi_tau(q: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Renyi exponent ``tau(q) = q h(q) - 1``."""
    return np.asarray(q, dtype=float) * np.asarray(h, dtype=float) - 1.0


def singularity_spectrum(
    q: Sequence[float] | np.ndarray, h: Sequence[float] | np.ndarray
) -> SpectrumResult:
    """Legendre transform of the Renyi exponent.

    ``alpha = d tau / d q`` is evaluated by second-order central differences on
    the supplied grid, and ``f(alpha) = q alpha - tau(q)``.

    The grid must be sorted and strictly increasing.
    """
    q = np.asarray(q, dtype=float)
    h = np.asarray(h, dtype=float)
    if q.size != h.size:
        raise ValueError("q and h must have the same length")
    if q.size < 3:
        raise ValueError("at least three moment orders are required")
    if not np.all(np.diff(q) > 0):
        raise ValueError("q must be strictly increasing")

    tau = renyi_tau(q, h)
    alpha = np.gradient(tau, q, edge_order=2)
    f = q * alpha - tau
    return SpectrumResult(q=q, h=h, tau=tau, alpha=alpha, f=f)


def spectrum_width(
    q: Sequence[float] | np.ndarray, h: Sequence[float] | np.ndarray
) -> float:
    """Convenience wrapper returning only ``Delta alpha``."""
    return singularity_spectrum(q, h).width


def anchored_fit(
    q: Sequence[float] | np.ndarray,
    h: Sequence[float] | np.ndarray,
    weights: np.ndarray | None = None,
) -> AnchoredFit:
    """Fit the anchored linear model for the generalised Hurst function.

    The model is anchored at the second moment, so ``H`` is read directly as
    ``h(2)`` rather than estimated, and only the slope is fitted:

        lambda_hat = sum_q c_q [h(q) - H] / sum_q (q - 2)^2

    with ``c_q = -(q - 2)`` by default.  The non-negativity-constrained variant
    ``lambda_plus = max(0, lambda_hat)`` is returned alongside, since the model
    restricts ``lambda >= 0``.

    ``r2`` is reported as the share of cross-``q`` variation in ``h`` explained
    by the anchored line, i.e. ``1 - sum(xi^2) / sum((h - mean(h))^2)``.

    Parameters
    ----------
    q
        Moment grid.  Must contain ``q = 2``.
    h
        Estimated generalised Hurst exponents on that grid.
    weights
        Optional override for ``c_q``.

    Returns
    -------
    AnchoredFit
    """
    q = np.asarray(q, dtype=float)
    h = np.asarray(h, dtype=float)
    if q.size != h.size:
        raise ValueError("q and h must have the same length")

    idx = np.flatnonzero(np.isclose(q, 2.0))
    if idx.size != 1:
        raise ValueError("the moment grid must contain q = 2 exactly once")
    big_h = float(h[idx[0]])

    qt = q - 2.0
    c = -qt if weights is None else np.asarray(weights, dtype=float)

    denom = float(np.sum(qt * qt))
    if denom <= 0:
        raise ValueError("degenerate moment grid: sum of (q - 2)^2 is zero")
    lam = float(np.sum(c * (h - big_h)) / denom)

    xi = h - (big_h - lam * qt)
    ss_tot = float(np.sum((h - h.mean()) ** 2))
    r2 = float(1.0 - np.sum(xi * xi) / ss_tot) if ss_tot > 0 else np.nan

    return AnchoredFit(
        H=big_h,
        lam=lam,
        lam_plus=max(0.0, lam),
        xi=xi,
        r2=r2,
        q=q,
    )
