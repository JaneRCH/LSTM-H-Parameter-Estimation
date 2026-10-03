"""Synthetic process families with known generalised Hurst functions.

Four families, each supplying an ``h(q)`` label that is known in closed form:

======================  =======================================================
Family                  ``h(q)``
======================  =======================================================
Fractional Brownian      ``H`` (constant): the monofractal null
Binomial cascade         ``[1 - log2(p^q + (1-p)^q)] / q``
Lognormal MMAR           ``Hb (1 + lc2/2) - (lc2 Hb^2 / 2) q``: linear in ``q``
Rough heavy-tailed       ``H`` for ``q <= gamma``; ``1/q + H - 1/gamma`` beyond
======================  =======================================================

The lognormal family is the one the supervisor identified as best matching the
empirical findings, and the reason is visible in the table: its ``h(q)`` is
exactly linear in ``q``, which is the anchored model ``h(q) = H - lambda*(q-2)``
of :mod:`rvmf.ghe` with

    H      = Hb (1 + lc2/2) - lc2 Hb^2,
    lambda = lc2 Hb^2 / 2.

Note the symbol collision this creates and which the report must resolve.  Here
``Hb`` is the Hurst parameter of the fractional Brownian *base* of the
subordination, ``lc2`` is the cascade intermittency, and ``H`` without a
subscript is reserved throughout for ``h(2)``, following the supervisor's
convention.

.. warning::
   The rough heavy-tailed label is inherited from the alpha-stable literature,
   where the tail index satisfies ``alpha < 2``.  The empirical JSE tail index
   is near three, which is outside that range and has finite variance, so the
   construction here filters Student-t innovations rather than stable ones and
   the piecewise label is **not** established for ``gamma > 2``.
   :func:`rvmf.validate.check_heavy_tail_label` measures it instead of assuming
   it.  Do not train on this label before that check has been read.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

__all__ = [
    "fbm",
    "fgn",
    "binomial_cascade",
    "lognormal_mmar",
    "rough_heavy_tailed",
    "h_fbm",
    "h_binomial",
    "h_lognormal_mmar",
    "h_rough_heavy_tailed",
    "mmar_moments",
    "FAMILIES",
]


# ----------------------------------------------------------------------------
# Fractional Brownian motion (exact, Davies-Harte circulant embedding)
# ----------------------------------------------------------------------------


def _fgn_autocovariance(n: int, hurst: float) -> np.ndarray:
    """Autocovariance of unit-variance fractional Gaussian noise at lags 0..n."""
    k = np.arange(n + 1, dtype=float)
    return 0.5 * (
        np.abs(k + 1) ** (2 * hurst)
        - 2.0 * np.abs(k) ** (2 * hurst)
        + np.abs(k - 1) ** (2 * hurst)
    )


def fgn(n: int, hurst: float, rng: np.random.Generator) -> np.ndarray:
    """Exact fractional Gaussian noise of unit variance.

    Uses the circulant-embedding method of Davies and Harte (1987), which is
    exact rather than approximate and runs in ``O(n log n)``.

    Parameters
    ----------
    n
        Number of increments.
    hurst
        Hurst parameter in ``(0, 1)``.
    rng
        Random generator.

    Returns
    -------
    ndarray, shape (n,)
    """
    if not 0.0 < hurst < 1.0:
        raise ValueError("hurst must lie in (0, 1)")
    if n < 2:
        raise ValueError("n must be at least 2")

    r = _fgn_autocovariance(n, hurst)
    # Circulant first row of size m = 2n.
    c = np.concatenate([r, r[-2:0:-1]])
    m = c.size
    lam = np.fft.fft(c).real

    # Negative eigenvalues indicate the embedding failed; clipping is the
    # standard remedy and is negligible when it occurs at all.
    if lam.min() < 0:
        lam = np.clip(lam, 0.0, None)

    w = rng.standard_normal(m) + 1j * rng.standard_normal(m)
    # Enforce the conjugate symmetry that makes the transform real.
    w[0] = np.sqrt(2.0) * w[0].real
    w[n] = np.sqrt(2.0) * w[n].real
    w[n + 1 :] = np.conj(w[1:n][::-1])

    z = np.fft.fft(np.sqrt(lam / (2.0 * m)) * w)
    return z[:n].real


def fbm(n: int, hurst: float, rng: np.random.Generator) -> np.ndarray:
    """Fractional Brownian motion path of length ``n``, starting at zero."""
    return np.concatenate([[0.0], np.cumsum(fgn(n, hurst, rng))])[: n]


def h_fbm(q: np.ndarray, hurst: float) -> np.ndarray:
    """Analytic ``h(q)`` of fBm: constant in ``q``."""
    return np.full_like(np.asarray(q, dtype=float), float(hurst))


# ----------------------------------------------------------------------------
# Binomial (p-model) cascade
# ----------------------------------------------------------------------------


def binomial_cascade(
    n_levels: int, p: float, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Deterministic binomial multiplicative measure on ``2 ** n_levels`` cells.

    The cascade is the ``p``-model of Meneveau and Sreenivasan (1987).  Mass is
    split in proportions ``p`` and ``1 - p`` at each level.  ``rng`` randomises
    which child receives ``p``; passing ``None`` gives the canonical ordering.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must lie in (0, 1)")
    mu = np.ones(1, dtype=float)
    for _ in range(n_levels):
        left = np.full(mu.size, p)
        if rng is not None:
            swap = rng.random(mu.size) < 0.5
            left = np.where(swap, 1.0 - p, p)
        mu = np.column_stack([mu * left, mu * (1.0 - left)]).ravel()
    return mu


def h_binomial(q: np.ndarray, p: float) -> np.ndarray:
    """Analytic ``h(q)`` of the binomial cascade.

    ``h(q) = [1 - log2(p^q + (1-p)^q)] / q``, with the ``q -> 0`` limit
    ``-log2(p(1-p)) / 2`` supplied explicitly.
    """
    q = np.asarray(q, dtype=float)
    out = np.empty_like(q)
    zero = np.isclose(q, 0.0)

    qn = q[~zero]
    out[~zero] = (1.0 - np.log2(p**qn + (1.0 - p) ** qn)) / qn
    out[zero] = -np.log2(p * (1.0 - p)) / 2.0
    return out


# ----------------------------------------------------------------------------
# Lognormal multifractal random walk in subordinated (MMAR) form
# ----------------------------------------------------------------------------


def _lognormal_trading_time(
    n_levels: int, lambda_c2: float, rng: np.random.Generator
) -> np.ndarray:
    """Increments of a lognormal cascade trading time, normalised to unit mass.

    The per-level lognormal multiplier is ``W = exp(sigma Z - sigma^2 / 2)``
    with ``sigma^2 = lambda_c2 * ln 2``, which is the variance that yields the
    cascade scaling ``h_theta(u) = 1 - (lambda_c2 / 2)(u - 1)``.
    """
    sigma2 = lambda_c2 * np.log(2.0)
    sigma = np.sqrt(sigma2)
    mu = np.ones(1, dtype=float)
    for _ in range(n_levels):
        w = np.exp(sigma * rng.standard_normal(2 * mu.size) - 0.5 * sigma2)
        mu = (np.repeat(mu, 2) * w)
    return mu / mu.sum()


def lognormal_mmar(
    n: int,
    hurst_base: float,
    lambda_c2: float,
    rng: np.random.Generator,
    n_levels: int | None = None,
    fine_power: int = 20,
    method: str = "compounding",
) -> np.ndarray:
    """Lognormal multifractal random walk, ``X(t) = B_Hb(theta(t))``.

    Parameters
    ----------
    n
        Length of the returned series.
    hurst_base
        Hurst parameter ``Hb`` of the fractional Brownian base.  This is *not*
        ``h(2)`` of the resulting process; see :func:`h_lognormal_mmar`.
    lambda_c2
        Cascade intermittency ``lambda_c^2``.  Zero recovers fBm.
    rng
        Random generator.
    n_levels
        Cascade depth.  Defaults to the smallest power of two covering ``n``.
    fine_power
        The base fBm is sampled on ``2 ** fine_power`` points and interpolated
        at the trading-time grid (``"subordination"`` only).
    method
        ``"compounding"`` (default) uses ``dX_t = mu_t ** Hb * g_t`` with ``g``
        unit fractional Gaussian noise, the standard scheme in the MMAR
        literature.  ``"subordination"`` evaluates the base path at the deformed
        time, which is the definition rather than an approximation.

        Subordination is *not* the default despite being exact in principle,
        because linear interpolation on a fixed fine grid smooths the path
        wherever a trading-time increment falls below one grid step, which an
        intermittent cascade produces in abundance.  Measured against the
        analytic label at ``Hb = 0.30``, ``lc2 = 0.40``, subordination attains a
        mean absolute error of 0.47, 0.34, 0.11 and 0.07 at fine grids of 2, 4,
        16 and 64 times the output length, so it converges only at a cost that
        is prohibitive corpus-wide.  Compounding attains 0.037 at no such cost.
        Both agree with fBm to within 0.01 at ``lc2 = 0``.

    Returns
    -------
    ndarray, shape (n,)
    """
    if lambda_c2 < 0:
        raise ValueError("lambda_c2 must be non-negative")
    levels = n_levels or int(np.ceil(np.log2(max(n, 2))))
    mu = _lognormal_trading_time(levels, lambda_c2, rng)

    if method == "compounding":
        g = fgn(mu.size, hurst_base, rng)
        increments = (mu ** hurst_base) * g
        path = np.cumsum(increments)
        return path[:n]

    if method != "subordination":
        raise ValueError("method must be 'subordination' or 'compounding'")

    theta = np.concatenate([[0.0], np.cumsum(mu)])
    n_fine = 2**fine_power
    base = np.concatenate([[0.0], np.cumsum(fgn(n_fine, hurst_base, rng))])
    # Self-similarity: a path on a unit interval sampled at n_fine steps has
    # increment scale n_fine ** -Hb.
    base = base * (n_fine ** (-hurst_base))
    grid = np.linspace(0.0, 1.0, n_fine + 1)
    path = np.interp(theta, grid, base)
    return path[:n]


def mmar_moments(hurst_base: float, lambda_c2: float) -> tuple[float, float]:
    """Return ``(H, lambda)`` of the anchored model for a lognormal MMAR.

    ``H`` is ``h(2)`` and ``lambda`` is the multifractality strength, i.e. the
    negative slope of ``h`` at ``q = 2``.
    """
    big_h = hurst_base * (1.0 + lambda_c2 / 2.0) - lambda_c2 * hurst_base**2
    lam = lambda_c2 * hurst_base**2 / 2.0
    return float(big_h), float(lam)


def h_lognormal_mmar(
    q: np.ndarray, hurst_base: float, lambda_c2: float
) -> np.ndarray:
    """Analytic ``h(q)`` of the subordinated lognormal MMAR.

    ``h(q) = Hb (1 + lc2 / 2) - (lc2 Hb^2 / 2) q``, exactly linear in ``q``.

    Values outside the range where the cascade has finite moments,
    ``|q * Hb| <= sqrt(2 / lc2)``, are returned as ``nan`` so that the training
    loss can be masked there rather than fitted to an extrapolation.
    """
    q = np.asarray(q, dtype=float)
    out = (
        hurst_base * (1.0 + lambda_c2 / 2.0)
        - 0.5 * lambda_c2 * hurst_base**2 * q
    )
    if lambda_c2 > 0:
        bound = np.sqrt(2.0 / lambda_c2)
        out = np.where(np.abs(q * hurst_base) <= bound, out, np.nan)
    return out


# ----------------------------------------------------------------------------
# Rough heavy-tailed process (the artefact null)
# ----------------------------------------------------------------------------


def rough_heavy_tailed(
    n: int,
    hurst: float,
    gamma: float,
    rng: np.random.Generator,
    n_lags: int | None = None,
) -> np.ndarray:
    """Fractionally filtered heavy-tailed noise, cumulated to a path.

    Innovations are Student-t with ``gamma`` degrees of freedom, so the tail
    index is ``gamma`` while the variance is finite for ``gamma > 2``.  They are
    convolved with the fractional-difference kernel
    ``b_j = (j+1)^d - j^d`` with ``d = hurst - 1/2``, which imparts second-moment
    scaling ``h(2) = hurst``.

    This is deliberately *not* a linear fractional stable motion.  A stable law
    requires a tail index below two, whereas the empirical JSE tail index is
    near three; the finite-variance construction is the one that matches the
    data.  The consequence is that the piecewise scaling label of
    :func:`h_rough_heavy_tailed` is not guaranteed, which is why it is measured.
    """
    if gamma <= 2.0:
        raise ValueError(
            "gamma must exceed 2 for the finite-variance construction; "
            "a stable-law simulator is required below that"
        )
    d = hurst - 0.5
    lags = n_lags or min(4 * n, 8192)

    # ARFIMA(0, d, 0) moving-average coefficients, built by the recursion
    #   b_0 = 1,   b_j = b_{j-1} (j - 1 + d) / j,
    # which is the Gamma-ratio kernel Gamma(j + d) / (Gamma(j + 1) Gamma(d))
    # evaluated stably.  It is well defined for negative d, unlike the
    # difference form (j + 1)^d - j^d, whose leading term diverges at j = 0.
    # Fractional integration of order d imparts H = d + 1/2 to finite-variance
    # innovations.
    j = np.arange(1, lags, dtype=float)
    kernel = np.concatenate([[1.0], np.cumprod((j - 1.0 + d) / j)])

    innov = rng.standard_t(df=gamma, size=n + lags)
    # Scale to unit variance so that paths are comparable across gamma.
    innov /= np.sqrt(gamma / (gamma - 2.0))

    filtered = np.convolve(innov, kernel, mode="valid")[:n]
    return np.cumsum(filtered)


def h_rough_heavy_tailed(
    q: np.ndarray, hurst: float, gamma: float
) -> np.ndarray:
    """Piecewise scaling form for a rough heavy-tailed process.

    ``h(q) = hurst`` for ``q <= gamma`` and ``1/q + hurst - 1/gamma`` beyond.

    .. warning::
       **This form does not describe the finite-variance process and must not be
       used as a supervised label.**  It is established for alpha-stable
       processes, whose tail index is below two.  Measured against simulation at
       ``gamma > 2`` it fails in the specific way that matters: the estimated
       ``h(q)`` declines *smoothly from ``q = 2`` onwards*, with no break at
       ``q = gamma``, whereas the piecewise form predicts a flat profile up to
       ``gamma`` and a kink there.

       At ``H = 0.25``, ``gamma = 3``, ``T = 8192`` over twelve replicates the
       estimate runs 0.296, 0.267, 0.231, 0.195 at ``q = 2, 3, 4, 5`` against a
       predicted 0.25, 0.25, 0.167, 0.117.  Mean absolute error against the
       piecewise form is 0.093, against a flat line 0.086: the piecewise form
       fits no better than assuming no structure at all.  The same holds at
       ``gamma`` of 2.5 and 4.

    The consequence for the design is constructive rather than destructive.  The
    process remains a necessary null, because it fabricates a declining ``h(q)``
    while containing no multifractality.  But the null must be imposed by
    *simulating* the process and comparing widths by Monte Carlo, not by
    overlaying this analytic profile parameterised by a measured tail index.
    The function is retained so that the contrast can be plotted.
    """
    q = np.asarray(q, dtype=float)
    out = np.full_like(q, float(hurst))
    beyond = q > gamma
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(beyond, 1.0 / q + hurst - 1.0 / gamma, out)
    return out


# ----------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Family:
    """A synthetic family: how to simulate it and what its label is."""

    name: str
    simulate: Callable[..., np.ndarray]
    label: Callable[..., np.ndarray]
    exact_label: bool


FAMILIES: dict[str, Family] = {
    "fbm": Family("fbm", fbm, h_fbm, True),
    "binomial": Family("binomial", binomial_cascade, h_binomial, True),
    "lognormal_mmar": Family("lognormal_mmar", lognormal_mmar, h_lognormal_mmar, True),
    "rough_heavy_tailed": Family(
        "rough_heavy_tailed", rough_heavy_tailed, h_rough_heavy_tailed, False
    ),
}
