"""Surrogate series for decomposing apparent multifractality.

The observed spectrum width decomposes into a distributional component, a
linear-correlation component and a nonlinear-correlation component,

    Delta_alpha = Delta_PDF + Delta_LM + Delta_NL,

and only the last is evidence of intrinsic multifractality, which requires a
nonlinear mechanism.  Two surrogates separate them:

``shuffle``
    A random permutation destroys all temporal dependence, linear and
    nonlinear, while preserving the marginal distribution exactly.  Width that
    survives shuffling is carried by the marginal.
``iaaft``
    The iterative amplitude-adjusted Fourier transform preserves both the
    marginal and the power spectrum, hence all linear correlation, while
    destroying nonlinear dependence.  Width that the IAAFT surrogate fails to
    reproduce is the nonlinear component.

Reading the supervisor's SPX exhibit against this decomposition: the IAAFT
surrogate reproduces the original ``h(2)`` almost exactly (0.181 against 0.185),
which places the nonlinear component near zero, while the shuffled surrogate
collapses the level (0.035) without flattening the profile.  If that pattern
repeats on JSE data it is a substantive result, and it runs against the
nonlinear-correlation attribution of the published JSE benchmark.

References
----------
Schreiber, T. and Schmitz, A. (1996). Improved surrogate data for nonlinearity
tests. Physical Review Letters 77(4), 635-638.
Venema, V., Ament, F. and Simmer, C. (2006). A stochastic iterative amplitude
adjusted Fourier transform algorithm with improved accuracy. Nonlinear
Processes in Geophysics 13(3), 321-328.
"""

from __future__ import annotations

import numpy as np

__all__ = ["shuffle_surrogate", "iaaft_surrogate", "surrogate_ensemble"]


def shuffle_surrogate(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random permutation: preserves the marginal, destroys all dependence."""
    return rng.permutation(np.asarray(x, dtype=float))


def iaaft_surrogate(
    x: np.ndarray,
    rng: np.random.Generator,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> np.ndarray:
    """IAAFT surrogate: preserves the marginal and the power spectrum.

    The iteration alternates two projections, one onto the set of series with
    the target amplitude spectrum and one onto the set with the target marginal,
    until the ranks stop changing.

    Parameters
    ----------
    x
        Source series.
    rng
        Random generator, used for the initial permutation.
    max_iter
        Maximum number of alternating projections.
    tol
        Convergence tolerance on the relative change in the amplitude spectrum.

    Returns
    -------
    ndarray
        A surrogate of the same length as ``x``.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    sorted_values = np.sort(x)
    target_amplitude = np.abs(np.fft.rfft(x))

    current = rng.permutation(x)
    previous_error = np.inf

    for _ in range(max_iter):
        # Projection 1: impose the target amplitude spectrum, keep the phases.
        spectrum = np.fft.rfft(current)
        phases = np.angle(spectrum)
        current = np.fft.irfft(target_amplitude * np.exp(1j * phases), n=n)

        # Projection 2: impose the target marginal by rank ordering.
        ranks = np.argsort(np.argsort(current))
        current = sorted_values[ranks]

        error = float(
            np.linalg.norm(np.abs(np.fft.rfft(current)) - target_amplitude)
            / np.linalg.norm(target_amplitude)
        )
        if abs(previous_error - error) < tol:
            break
        previous_error = error

    return current


def surrogate_ensemble(
    x: np.ndarray,
    kind: str,
    n_surrogates: int,
    rng: np.random.Generator,
    **kwargs,
) -> np.ndarray:
    """Generate ``n_surrogates`` surrogates of the given kind.

    Returns
    -------
    ndarray, shape (n_surrogates, len(x))
    """
    makers = {"shuffle": shuffle_surrogate, "iaaft": iaaft_surrogate}
    if kind not in makers:
        raise ValueError(f"kind must be one of {sorted(makers)}")
    maker = makers[kind]
    return np.array([maker(x, rng, **kwargs) for _ in range(n_surrogates)])
