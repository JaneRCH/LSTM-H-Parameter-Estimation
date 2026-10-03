r"""Training corpus: labelled daily bars from the log S-fBM observation model.

Each draw fixes a latent process and a price-recording process, simulates daily
bars, and labels them with the latent :math:`(H, \lambda^2)`.  The label is a
property of the generating process, never an exponent measured on the proxy.
That is the change this redesign turns on: the interim corpus labelled each
path with the scaling exponent of the path itself and fed the network a clean
series, so the network never met the measurement noise that dominates the real
data.

Priors follow the design note.  Mass is placed near :math:`H = 0` so that the
multifractal end of the continuum is learnt rather than extrapolated, and the
nuisance parameters span the range the JSE files plausibly occupy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features_v2 import FeatureSpecV2, extract
from .rangeproxy import log_proxy
from .observation import ObservationParams, simulate_bars
from .sfbm import SfbmParams

__all__ = ["PriorSpec", "draw_parameters", "simulate_labelled", "build_corpus"]


@dataclass(frozen=True)
class PriorSpec:
    """Prior over the target and nuisance parameters."""

    near_zero_share: float = 0.25
    H_near_zero: tuple[float, float] = (0.0, 0.03)
    H_bulk: tuple[float, float] = (0.03, 0.49)
    lambda_sq: tuple[float, float] = (0.005, 0.20)
    T_over_L: tuple[float, float] = (1.0, 10.0)
    sigma_bar: tuple[float, float] = (0.12, 0.35)
    rho: tuple[float, float] = (-0.8, 0.0)
    kappa: tuple[float, float] = (0.0, 0.3)
    student_share: float = 0.5
    student_df: tuple[float, float] = (4.0, 30.0)
    steps_per_day: int = 48


def draw_parameters(
    length: int, prior: PriorSpec, rng: np.random.Generator
) -> tuple[SfbmParams, ObservationParams]:
    """Draw one latent process and one observation process from the prior."""
    if rng.random() < prior.near_zero_share:
        H = float(rng.uniform(*prior.H_near_zero))
    else:
        H = float(rng.uniform(*prior.H_bulk))

    lam = float(np.exp(rng.uniform(*np.log(prior.lambda_sq))))
    t_over_l = float(np.exp(rng.uniform(*np.log(prior.T_over_L))))
    df = float(rng.uniform(*prior.student_df)) if rng.random() < prior.student_share else None

    sf = SfbmParams(
        H=H,
        lambda_sq=lam,
        T=t_over_l * length,
        sigma_bar=float(rng.uniform(*prior.sigma_bar)),
    )
    obs = ObservationParams(
        rho=float(rng.uniform(*prior.rho)),
        kappa=float(rng.uniform(*prior.kappa)),
        df=df,
        steps_per_day=prior.steps_per_day,
    )
    return sf, obs


def simulate_labelled(
    length: int,
    prior: PriorSpec,
    spec: FeatureSpecV2,
    rng: np.random.Generator,
    with_proxy: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Simulate one labelled example.

    Returns the sequence branch, the summary branch, the label
    ``(H, log lambda^2)`` and, when ``with_proxy`` is set, the primary log
    proxy series itself.

    The raw series is kept for held-out corpora only.  The classical
    estimators consume the series rather than the features, and scoring them on
    the same paths as the network is what makes the comparison like for like.
    Storing it for the training corpus as well would cost several hundred
    megabytes for no use.
    """
    sf, obs = draw_parameters(length, prior, rng)
    bars = simulate_bars(length, sf, obs, rng)
    seq, summ = extract(bars, spec)
    label = np.array([sf.H, np.log(sf.lambda_sq)], dtype=np.float32)
    proxy = log_proxy(bars, spec.primary).astype(np.float32) if with_proxy else None
    return seq, summ, label, proxy


def build_corpus(
    n: int,
    length: int,
    prior: PriorSpec,
    spec: FeatureSpecV2,
    rng: np.random.Generator,
    progress: int = 0,
    with_proxy: bool = False,
) -> dict[str, np.ndarray]:
    """Simulate ``n`` labelled examples.

    The corpus is pre-generated rather than produced fresh each epoch, which
    departs from :cite:`Boros2024`.  Simulating the full observation model costs
    about a tenth of a second per path on the available hardware, so on-the-fly
    training at their scale would spend hours per network in simulation alone.
    The departure is controlled rather than assumed away: the held-out corpus is
    drawn from an independent seed stream, and the gap between training and
    held-out loss is reported at every evaluation, so overfitting would be
    visible rather than hidden.
    """
    seqs, summs, labels, proxies = [], [], [], []
    for i in range(n):
        seq, summ, label, proxy = simulate_labelled(
            length, prior, spec, rng, with_proxy=with_proxy
        )
        seqs.append(seq)
        summs.append(summ)
        labels.append(label)
        if with_proxy:
            proxies.append(proxy)
        if progress and (i + 1) % progress == 0:
            print(f"  [corpus] {i + 1}/{n}", flush=True)
    out = {
        "sequence": np.stack(seqs),
        "summary": np.stack(summs),
        "label": np.stack(labels),
    }
    if with_proxy:
        out["proxy"] = np.stack(proxies)
    return out
