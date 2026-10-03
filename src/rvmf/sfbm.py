r"""Log S-fBM: one family spanning rough and multifractal volatility.

The stationary fractional Brownian motion of :cite:`WuMuzyBacry2022` is the
zero-mean Gaussian process :math:`\omega_{H,T}` with autocovariance

.. math::
    \operatorname{Cov}(\omega_s, \omega_{s+\tau})
      = \frac{\nu^2}{2}\left(T^{2H} - \tau^{2H}\right), \qquad \tau < T,

and zero beyond the correlation scale :math:`T`.  The intermittency
coefficient is :math:`\lambda^2 = H(1-2H)\nu^2`.  Holding :math:`\lambda^2`
fixed and letting :math:`H \to 0` gives

.. math::
    \operatorname{Cov}(\omega_s, \omega_{s+\tau}) \to \lambda^2 \ln(T/\tau),

the log-normal multifractal random measure.  One parameter therefore moves the
process along a continuum from rough volatility (:math:`0 < H < 1/2`) to
multifractality (:math:`H = 0`), which is what the title of this study claims
to traverse.

Two practical points.

The variance at :math:`\tau = 0` diverges as :math:`H \to 0`, so a small-scale
cutoff is needed.  It must sit *strictly below* the simulation grid.  Placing it
at the grid step itself gives :math:`C(\delta) = C(0)`, which forces
:math:`\operatorname{Var}(\omega_{t+\delta} - \omega_t) = 0`: the process
becomes degenerate at the finest resolved scale, and whatever variation the
simulator produces there is an artefact of eigenvalue clipping rather than a
property of the model.

This module therefore evaluates the covariance at :math:`\tau_k = k\delta` for
:math:`k \geq 1` and at :math:`\tau_0 = \delta / e`.  The choice of
:math:`e` fixes the scale of the cutoff relative to the grid and is a declared
constant of the discretisation, not a fitted quantity.  It gives a lag-one
increment variance of exactly :math:`2\lambda^2` in the multifractal limit,
since :math:`2\lambda^2 \ln(\delta / \tau_0) = 2\lambda^2`, and it leaves the
covariance finite and continuous in :math:`H` through zero.  The latent process
is simulated on an intraday grid, so the cutoff corresponds to roughly one part
in a hundred and thirty of a trading day.

The resulting structure function is

.. math::
    \mathbb{E}\left[(\omega_{t+\tau} - \omega_t)^2\right]
      = \nu^2 \left(\tau^{2H} - \tau_0^{2H}\right),

free of :math:`T` and of the sample mean, which is what makes it the right
quantity against which to validate the simulator; see
Appendix~:ref:`app:validation`.

Only :math:`(H, \lambda^2)` are targeted.  :cite:`WuMuzyBacry2022` show that
:math:`\nu^2` is poorly identified when :math:`H` is small, because
:math:`\hat\nu^2 \approx \lambda^2 / [\hat H (1 - 2\hat H)]` inherits the whole
error of :math:`\hat H` through a diverging factor.  :math:`T` is a nuisance
parameter and is not identifiable when the sample is shorter than :math:`T`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["CUTOFF_RATIO", "SfbmParams", "autocovariance", "increment_variance",
           "nu_squared", "simulate_omega", "spot_variance"]

#: Small-scale cutoff as a fraction of the simulation step.  Must be < 1; see the
#: module docstring for why the cutoff cannot sit at the grid step itself.
CUTOFF_RATIO: float = 1.0 / np.e


@dataclass(frozen=True)
class SfbmParams:
    """Parameters of a log S-fBM volatility process.

    Attributes
    ----------
    H
        Roughness, in ``[0, 0.5)``.  ``H = 0`` is the multifractal limit.
    lambda_sq
        Intermittency coefficient, strictly positive.
    T
        Correlation scale in days.
    sigma_bar
        Annualised volatility level; the spot variance has mean
        ``sigma_bar**2`` per year.
    """

    H: float
    lambda_sq: float
    T: float
    sigma_bar: float = 0.20

    def __post_init__(self) -> None:
        if not (0.0 <= self.H < 0.5):
            raise ValueError(f"H must lie in [0, 0.5), got {self.H}")
        if self.lambda_sq <= 0.0:
            raise ValueError(f"lambda_sq must be positive, got {self.lambda_sq}")
        if self.T <= 0.0:
            raise ValueError(f"T must be positive, got {self.T}")
        if self.sigma_bar <= 0.0:
            raise ValueError(f"sigma_bar must be positive, got {self.sigma_bar}")


def nu_squared(H: float, lambda_sq: float) -> float:
    """Return ``nu^2 = lambda^2 / (H(1 - 2H))``, the unnormalised amplitude.

    Diverges as ``H -> 0``; reported only for completeness and never estimated.
    """
    if H <= 0.0:
        return float("inf")
    return float(lambda_sq / (H * (1.0 - 2.0 * H)))


def autocovariance(n: int, delta: float, p: SfbmParams) -> np.ndarray:
    """Autocovariance of ``omega`` at lags ``0, delta, ..., (n-1) delta``.

    The zero lag is evaluated at ``delta * CUTOFF_RATIO`` rather than at zero,
    so that the ``H -> 0`` limit stays finite.  The cutoff sits strictly below
    the grid step; see the module docstring for why that matters.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    if delta <= 0.0:
        raise ValueError("delta must be positive")

    tau = np.arange(n, dtype=float) * delta
    tau[0] = delta * CUTOFF_RATIO
    if p.H == 0.0:
        cov = p.lambda_sq * np.log(p.T / tau)
    else:
        amp = nu_squared(p.H, p.lambda_sq) / 2.0
        cov = amp * (p.T ** (2.0 * p.H) - tau ** (2.0 * p.H))
    return np.maximum(cov, 0.0) * (tau < p.T)


def increment_variance(tau: np.ndarray, delta: float, p: SfbmParams) -> np.ndarray:
    r"""Analytic :math:`\mathbb{E}[(\omega_{t+\tau} - \omega_t)^2]`.

    Equals :math:`\nu^2(\tau^{2H} - \tau_0^{2H})` for ``H > 0`` and
    :math:`2\lambda^2 \ln(\tau / \tau_0)` at ``H = 0``, where
    :math:`\tau_0 = \delta \cdot` :data:`CUTOFF_RATIO`.  Free of ``T`` and of
    the sample mean, so it is well defined on a window shorter than ``T``.  This
    is the validation target for :func:`simulate_omega`.
    """
    tau = np.asarray(tau, dtype=float)
    tau0 = delta * CUTOFF_RATIO
    if p.H == 0.0:
        return 2.0 * p.lambda_sq * np.log(tau / tau0)
    return nu_squared(p.H, p.lambda_sq) * (tau ** (2.0 * p.H) - tau0 ** (2.0 * p.H))


def simulate_omega(
    n: int, delta: float, p: SfbmParams, rng: np.random.Generator
) -> np.ndarray:
    """Simulate ``n`` points of ``omega`` by Davies-Harte circulant embedding.

    The embedding is exact when every eigenvalue of the circulant extension is
    non-negative.  The size is doubled until that holds, up to
    :data:`_MAX_EMBED_DOUBLINGS`; any residual negative mass is clipped and is
    reported by :func:`embedding_defect`.
    """
    lam, size = _embedding(n, delta, p)
    lam = np.maximum(lam, 0.0)
    assert lam.size == size
    # Re and Im of the transform are two independent realisations, each with
    # the target covariance. The complex noise is left unscaled so that the
    # real part carries the full variance rather than half of it.
    noise = rng.standard_normal(size) + 1j * rng.standard_normal(size)
    out = np.fft.fft(np.sqrt(lam) * noise) / np.sqrt(size)
    return np.real(out[:n])


def embedding_defect(n: int, delta: float, p: SfbmParams) -> float:
    """Share of circulant eigenvalue mass clipped to zero; zero means exact."""
    lam, _ = _embedding(n, delta, p)
    negative = -float(np.sum(lam[lam < 0.0]))
    total = float(np.sum(np.abs(lam)))
    return negative / total if total > 0.0 else 0.0


_MAX_EMBED_DOUBLINGS = 3
_EMBED_TOL = 5e-2


def _embedding(n: int, delta: float, p: SfbmParams) -> tuple[np.ndarray, int]:
    """Circulant eigenvalues for an ``n``-point sample, and the circulant size.

    The covariance is evaluated out to lag ``size / 2`` rather than zero-padded.
    Zero padding would be wrong whenever the covariance has not decayed by lag
    ``n``, which is the usual case here because the correlation scale ``T`` is
    longer than the sample.
    """
    size = 1
    while size < 2 * n:
        size *= 2
    for _ in range(_MAX_EMBED_DOUBLINGS + 1):
        half = size // 2
        cov = autocovariance(half + 1, delta, p)
        first_row = np.concatenate([cov, cov[-2:0:-1]])
        lam = np.real(np.fft.fft(first_row))
        # A covariance truncated at T is only approximately embeddable, so a
        # handful of eigenvalues are always slightly negative. What matters is
        # the share of spectral mass they carry, not their sign: below
        # _EMBED_TOL the clipped simulation is exact to machine-level accuracy
        # and doubling the circulant only costs time.
        negative = -float(np.sum(lam[lam < 0.0]))
        total = float(np.sum(np.abs(lam)))
        if total <= 0.0 or negative / total <= _EMBED_TOL:
            return lam, lam.size
        size *= 2
    return lam, lam.size


def spot_variance(omega: np.ndarray, p: SfbmParams, trading_days: int = 252) -> np.ndarray:
    """Convert ``omega`` to a spot variance per day with mean ``sigma_bar^2/252``.

    The exponential is centred so that ``E[sigma^2]`` equals the target level
    exactly, rather than approximately: for a Gaussian ``omega``,
    ``E[exp(omega)] = exp(Var(omega)/2)``.
    """
    var_omega = float(np.var(omega))
    daily = (p.sigma_bar ** 2) / float(trading_days)
    return daily * np.exp(omega - 0.5 * var_omega)
