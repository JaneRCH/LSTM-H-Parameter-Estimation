"""Monte Carlo tests of an observed multifractal width against explicit nulls.

The natural test, that an estimated ``Delta alpha > 0`` demonstrates
multifractality, is invalid at finite sample length: a strictly monofractal
series of finite length yields a positive width by construction.  The remedy is
not to abandon the statistic but to simulate its null distribution at the length,
moment grid and scaling policy actually used, so that any bias shared by the
observed series and the null replicates cancels in the comparison.

Three nulls are supported.

``monofractal``
    Fractional Gaussian noise at the estimated ``h(2)``.  Rejection excludes the
    finite-size artefact channel.
``heavy_tailed``
    The rough heavy-tailed process at the estimated ``h(2)`` and measured tail
    index.  This null is imposed by *simulation*, not by overlaying the
    alpha-stable piecewise profile, because that profile does not describe the
    process at the tail indices the data exhibit; see
    :func:`rvmf.simulators.h_rough_heavy_tailed`.
``shuffle``
    Random permutations of the observed series, preserving the marginal exactly
    and destroying all dependence.  Rejection excludes the distributional
    channel.

Genuine multifractality requires all three to be rejected.  Requiring every
component to reject is an intersection-union test, whose size is bounded by the
size of each component, so the procedure remains valid at the nominal level
without a multiplicity correction across the three.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

from .ghe import spectrum_width
from .mfdfa import DEFAULT_Q, ScalingRange, mfdfa
from .simulators import fgn, rough_heavy_tailed
from .surrogates import shuffle_surrogate

__all__ = ["NullResult", "monte_carlo_test"]

NullKind = Literal["monofractal", "heavy_tailed", "shuffle"]


@dataclass
class NullResult:
    """Outcome of a Monte Carlo comparison against one null."""

    kind: str
    observed: float
    null_widths: np.ndarray
    p_value: float
    n_replicates: int

    def summary(self) -> dict:
        return {
            "null": self.kind,
            "observed_width": self.observed,
            "null_mean": float(np.mean(self.null_widths)),
            "null_sd": float(np.std(self.null_widths)),
            "null_p95": float(np.percentile(self.null_widths, 95)),
            "p_value": self.p_value,
            "n_replicates": self.n_replicates,
        }


def _width_of(
    x: np.ndarray,
    q: np.ndarray,
    order: int,
    spec: ScalingRange,
    estimator: Callable[[np.ndarray], np.ndarray] | None,
) -> float:
    h = estimator(x) if estimator is not None else mfdfa(x, q, order, spec).h
    return spectrum_width(q, h)


def monte_carlo_test(
    x: np.ndarray,
    kind: NullKind,
    n_replicates: int = 1000,
    rng: np.random.Generator | None = None,
    q: np.ndarray = DEFAULT_Q,
    order: int = 2,
    scaling_range: ScalingRange | None = None,
    estimator: Callable[[np.ndarray], np.ndarray] | None = None,
    tail_index: float | None = None,
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> NullResult:
    """Compare the observed spectrum width against a simulated null.

    Parameters
    ----------
    x
        Observed estimand.
    kind
        Which null to impose.
    n_replicates
        Number of null realisations.
    rng
        Random generator.
    q, order, scaling_range
        Estimation settings, applied identically to observation and replicates.
    estimator
        Optional callable mapping a series to ``h(q)``, used to run the same
        test under the learned estimator.  Defaults to MF-DFA.
    tail_index
        Required for the ``heavy_tailed`` null.
    transform
        Optional map applied to each null replicate before estimation, for
        example the rolling window of a range-based proxy.  Passing the same
        smoothing the observed series underwent keeps the induced
        autocorrelation in the null and the observation alike, so it cancels.

    Returns
    -------
    NullResult
    """
    rng = rng or np.random.default_rng()
    spec = scaling_range or ScalingRange()
    x = np.asarray(x, dtype=float)
    n = x.size

    observed_h = estimator(x) if estimator is not None else mfdfa(x, q, order, spec).h
    observed = spectrum_width(q, observed_h)
    h2 = float(observed_h[int(np.argmin(np.abs(q - 2.0)))])

    widths = np.empty(n_replicates)
    for i in range(n_replicates):
        if kind == "monofractal":
            # Clip into the open unit interval: a level-convention estimate can
            # land outside the fGn parameter range.
            rep = fgn(n, float(np.clip(h2, 0.02, 0.98)), rng)
        elif kind == "heavy_tailed":
            if tail_index is None:
                raise ValueError("tail_index is required for the heavy_tailed null")
            rep = np.diff(
                rough_heavy_tailed(
                    n + 1, float(np.clip(h2, 0.02, 0.98)), float(tail_index), rng
                )
            )
        elif kind == "shuffle":
            rep = shuffle_surrogate(x, rng)
        else:
            raise ValueError(f"unknown null {kind!r}")

        if transform is not None:
            rep = transform(rep)
        widths[i] = _width_of(rep, q, order, spec, estimator)

    # Proportion of replicates at least as extreme, with the usual
    # add-one correction so the p-value is never exactly zero.
    p = float((np.sum(widths >= observed) + 1) / (n_replicates + 1))
    return NullResult(
        kind=kind,
        observed=observed,
        null_widths=widths,
        p_value=p,
        n_replicates=n_replicates,
    )
