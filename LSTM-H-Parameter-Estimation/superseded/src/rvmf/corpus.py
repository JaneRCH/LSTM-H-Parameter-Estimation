"""Synthetic corpus: parameter sampling, simulation and labelling.

Batches are simulated fresh rather than drawn from a stored training set, so no
fixed corpus exists to overfit.  Only the held-out evaluation sets are fixed, by
seed, so that every reported number is reproducible.

Family shares default to a lognormal-weighted mix.  The lognormal multifractal
random walk carries the largest share because its ``h(q)`` is exactly linear in
``q``, which is the anchored model the empirical profiles follow; fractional
Brownian motion supplies the monofractal null; and the binomial cascade, whose
geometry resembles a volatility series least, is retained only to anchor the
strongly curved end of the label range.

Each simulated series is fed to the estimator in the convention under which its
analytic label holds, namely as a noise-like series rather than a path.  For
fractional Brownian motion that is the fractional Gaussian noise; for the
lognormal family it is the increment series; for the cascade it is the measure
itself.  Mixing the two conventions is what makes a label look wrong by exactly
one, as the ``+1`` check in :mod:`rvmf.proxies` shows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .features import FeatureSpec, extract_features
from .simulators import (
    binomial_cascade,
    fgn,
    h_binomial,
    h_fbm,
    h_lognormal_mmar,
    h_rough_heavy_tailed,
    lognormal_mmar,
    mmar_moments,
    rough_heavy_tailed,
)

__all__ = [
    "CorpusSpec",
    "sample_path",
    "make_batch",
    "fixed_testset",
    "parse_shares",
    "DEFAULT_Q",
    "DEFAULT_SHARES",
]

DEFAULT_Q = np.arange(-5, 6, dtype=float)

DEFAULT_SHARES: dict[str, float] = {
    "fbm": 0.30,
    "lognormal_mmar": 0.50,
    "binomial": 0.20,
    "rough_heavy_tailed": 0.0,
}


def parse_shares(text: str | None) -> dict[str, float]:
    """Parse a ``family=weight,...`` mixture override.

    Returns the default mixture unchanged when ``text`` is ``None``, so a
    caller can pass an unset command-line flag straight through. A family
    absent from the string is set to zero, which is what makes dropping a
    family a one-flag change rather than an edit to the source. Weights are
    renormalised by :meth:`CorpusSpec.families`, so they need not sum to one.

    Raises
    ------
    ValueError
        If a family name is unknown, a weight is not a finite number, a weight
        is negative, or every weight is zero.
    """
    if text is None:
        return dict(DEFAULT_SHARES)
    shares = {k: 0.0 for k in DEFAULT_SHARES}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"expected 'family=weight', got {item!r}")
        name, _, weight = item.partition("=")
        name = name.strip()
        if name not in shares:
            raise ValueError(
                f"unknown family {name!r}; known: {sorted(shares)}"
            )
        value = float(weight)
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"weight for {name!r} must be finite and >= 0")
        shares[name] = value
    if sum(shares.values()) <= 0:
        raise ValueError("at least one family must carry a positive weight")
    return shares


@dataclass(frozen=True)
class CorpusSpec:
    """Sampling design for the synthetic corpus."""

    shares: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_SHARES))
    length: int = 5000
    # fBm
    fbm_h: tuple[float, float] = (0.05, 0.95)
    # Lognormal MMAR base Hurst and cascade intermittency
    mmar_hb: tuple[float, float] = (0.08, 0.55)
    mmar_lc2: tuple[float, float] = (0.10, 0.70)
    # Binomial cascade
    binomial_p: tuple[float, float] = (0.50, 0.90)
    # Rough heavy-tailed (disabled by default; label not established, see
    # rvmf.simulators)
    rht_h: tuple[float, float] = (0.05, 0.50)
    rht_gamma: tuple[float, float] = (2.2, 5.0)
    # Oversampling band on h(2), applied by rejection
    band: tuple[float, float] = (0.10, 0.35)
    band_weight: float = 0.45

    def families(self) -> tuple[list[str], np.ndarray]:
        items = [(k, v) for k, v in self.shares.items() if v > 0]
        names = [k for k, _ in items]
        probs = np.array([v for _, v in items], dtype=float)
        return names, probs / probs.sum()


def _draw(rng: np.random.Generator, lo_hi: tuple[float, float]) -> float:
    lo, hi = lo_hi
    return float(rng.uniform(lo, hi))


def _h2_of(family: str, params: dict) -> float:
    """Analytic ``h(2)`` implied by a family and its parameters."""
    if family == "fbm":
        return params["hurst"]
    if family == "lognormal_mmar":
        return mmar_moments(params["hurst_base"], params["lambda_c2"])[0]
    if family == "binomial":
        return float(h_binomial(np.array([2.0]), params["p"])[0])
    if family == "rough_heavy_tailed":
        return params["hurst"]
    raise ValueError(family)


def _sample_params(
    family: str, rng: np.random.Generator, spec: CorpusSpec
) -> dict:
    """Draw parameters, oversampling the band of interest by rejection."""
    lo, hi = spec.band
    for attempt in range(32):
        if family == "fbm":
            params = {"hurst": _draw(rng, spec.fbm_h)}
        elif family == "lognormal_mmar":
            params = {
                "hurst_base": _draw(rng, spec.mmar_hb),
                "lambda_c2": _draw(rng, spec.mmar_lc2),
            }
        elif family == "binomial":
            params = {"p": _draw(rng, spec.binomial_p)}
        elif family == "rough_heavy_tailed":
            params = {
                "hurst": _draw(rng, spec.rht_h),
                "gamma": _draw(rng, spec.rht_gamma),
            }
        else:
            raise ValueError(family)

        in_band = lo <= _h2_of(family, params) <= hi
        # Accept in-band draws always; accept out-of-band draws with the
        # complementary probability, which concentrates mass in the band
        # without ever making the rest of the range unreachable.
        if in_band or rng.random() > spec.band_weight:
            return params
    return params


def sample_path(
    rng: np.random.Generator,
    spec: CorpusSpec,
    q: np.ndarray = DEFAULT_Q,
    family: str | None = None,
) -> tuple[np.ndarray, np.ndarray, str, dict]:
    """Simulate one labelled series.

    Returns
    -------
    x
        The estimand, in the convention under which the label holds.
    label
        ``h(q)`` on the grid ``q``.  Entries outside the range where the label
        is valid are ``nan`` and must be masked in the loss.
    family
        Family name.
    params
        Sampled parameters.
    """
    if family is None:
        names, probs = spec.families()
        family = str(rng.choice(names, p=probs))
    params = _sample_params(family, rng, spec)
    n = spec.length

    if family == "fbm":
        x = fgn(n, params["hurst"], rng)
        label = h_fbm(q, params["hurst"])
    elif family == "lognormal_mmar":
        path = lognormal_mmar(
            n + 1, params["hurst_base"], params["lambda_c2"], rng
        )
        x = np.diff(path)
        label = h_lognormal_mmar(q, params["hurst_base"], params["lambda_c2"])
    elif family == "binomial":
        levels = int(np.ceil(np.log2(n)))
        x = binomial_cascade(levels, params["p"], rng)[:n]
        label = h_binomial(q, params["p"])
    elif family == "rough_heavy_tailed":
        path = rough_heavy_tailed(n + 1, params["hurst"], params["gamma"], rng)
        x = np.diff(path)
        label = h_rough_heavy_tailed(q, params["hurst"], params["gamma"])
    else:
        raise ValueError(family)

    return x[:n], np.asarray(label, dtype=float), family, params


def make_batch(
    rng: np.random.Generator,
    spec: CorpusSpec,
    fspec: FeatureSpec,
    size: int,
    q: np.ndarray = DEFAULT_Q,
    family: str | None = None,
    return_series: bool = False,
):
    """Simulate a batch of labelled feature sequences.

    Returns
    -------
    features : ndarray, shape (size, seq_len, n_features)
    labels   : ndarray, shape (size, len(q)), with ``nan`` replaced by zero
    mask     : ndarray, shape (size, len(q)), boolean, True where the label holds
    families : list of str
    params   : list of dict
    series   : ndarray, shape (size, length), only when ``return_series``

    The raw series is returned on request so that a classical benchmark can be
    computed on exactly the paths the network saw, rather than on paths
    regenerated from the same seed, which would silently depend on the two code
    paths consuming the generator identically.
    """
    feats, labels, fams, pars, raw = [], [], [], [], []
    for _ in range(size):
        x, lab, fam, par = sample_path(rng, spec, q, family=family)
        feats.append(extract_features(x, fspec))
        labels.append(lab)
        fams.append(fam)
        pars.append(par)
        if return_series:
            raw.append(x)

    f = np.stack(feats).astype(np.float32)
    y = np.stack(labels).astype(np.float32)
    m = np.isfinite(y)
    y = np.nan_to_num(y, nan=0.0)
    if return_series:
        return f, y, m, fams, pars, np.stack(raw)
    return f, y, m, fams, pars


def fixed_testset(
    seed: int,
    spec: CorpusSpec,
    fspec: FeatureSpec,
    size: int,
    q: np.ndarray = DEFAULT_Q,
    family: str | None = None,
    return_series: bool = False,
):
    """Deterministic held-out evaluation set."""
    return make_batch(
        np.random.default_rng(seed),
        spec,
        fspec,
        size,
        q,
        family=family,
        return_series=return_series,
    )
