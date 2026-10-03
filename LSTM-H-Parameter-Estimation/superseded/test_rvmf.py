"""Correctness tests for the estimation machinery.

These guard the properties the report's claims rest on: that the simulators
generate what they claim, that the estimator recovers known labels, that the
scaling-range policy cannot select an unsupported window, that the surrogates
have the invariances the decomposition assumes, and that the feature map does
not silently collapse.

Run:  PYTHONPATH=src python3 -m pytest tests -q
"""

from __future__ import annotations

import numpy as np
import pytest

from rvmf.features import FeatureSpec, extract_features, feature_groups, feature_names
from rvmf.ghe import anchored_fit, singularity_spectrum, spectrum_width
from rvmf.mfdfa import ScalingRange, mfdfa
from rvmf.simulators import (
    _fgn_autocovariance,
    binomial_cascade,
    fgn,
    h_binomial,
    h_lognormal_mmar,
    lognormal_mmar,
    mmar_moments,
    rough_heavy_tailed,
)
from rvmf.surrogates import iaaft_surrogate, shuffle_surrogate

Q = np.arange(-5, 6, dtype=float)
I2 = int(np.flatnonzero(Q == 2.0)[0])
SPEC = ScalingRange(policy="fixed", s_min=10, min_segments=10)


@pytest.fixture
def rng():
    return np.random.default_rng(20260919)


# ---------------------------------------------------------------- simulators


def test_fgn_has_unit_variance_and_theoretical_acf(rng):
    paths = np.array([fgn(4096, 0.3, rng) for _ in range(60)])
    assert paths.var(axis=1).mean() == pytest.approx(1.0, abs=0.02)

    centred = paths - paths.mean(axis=1, keepdims=True)
    theory = _fgn_autocovariance(4, 0.3)[1:4]
    for lag, expected in enumerate(theory, start=1):
        empirical = (centred[:, :-lag] * centred[:, lag:]).mean()
        assert empirical == pytest.approx(expected, abs=0.02)


def test_fgn_rejects_invalid_hurst(rng):
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            fgn(128, bad, rng)


def test_binomial_cascade_conserves_mass(rng):
    mu = binomial_cascade(10, 0.7, rng)
    assert mu.size == 2**10
    assert mu.sum() == pytest.approx(1.0)
    assert np.all(mu > 0)


def test_binomial_label_is_flat_at_the_monofractal_point():
    h = h_binomial(Q, 0.5)
    assert np.allclose(h, 1.0)


def test_mmar_reduces_to_fbm_when_intermittency_is_zero(rng):
    x = np.diff(lognormal_mmar(2**13 + 1, 0.30, 0.0, rng))
    h = mfdfa(x, Q, 2, SPEC).h
    assert h[I2] == pytest.approx(0.30, abs=0.05)
    # A zero-intermittency cascade is monofractal, so the profile is flat.
    assert h.max() - h.min() < 0.25


def test_mmar_moments_match_the_analytic_label():
    for hb, lc2 in ((0.30, 0.40), (0.25, 0.55)):
        big_h, lam = mmar_moments(hb, lc2)
        label = h_lognormal_mmar(Q, hb, lc2)
        assert label[I2] == pytest.approx(big_h)
        # lambda is the negative slope of a line that is exactly linear in q.
        slope = (label[I2 + 1] - label[I2]) / 1.0
        assert -slope == pytest.approx(lam)


def test_rough_heavy_tailed_requires_finite_variance(rng):
    with pytest.raises(ValueError):
        rough_heavy_tailed(512, 0.25, 1.8, rng)


def test_rough_heavy_tailed_recovers_hurst_with_light_tails(rng):
    # Very high degrees of freedom is effectively Gaussian, so h(2) = H.
    x = np.diff(rough_heavy_tailed(2**13 + 1, 0.30, 200.0, rng))
    assert mfdfa(x, Q, 2, SPEC).h[I2] == pytest.approx(0.30, abs=0.06)


# ---------------------------------------------------------------------- mfdfa


def test_mfdfa_recovers_the_fgn_exponent(rng):
    for hurst in (0.25, 0.50, 0.70):
        h = mfdfa(fgn(2**13, hurst, rng), Q, 2, SPEC).h
        assert h[I2] == pytest.approx(hurst, abs=0.06)


def test_path_input_returns_one_more_than_series_input(rng):
    noise = fgn(2**13, 0.30, rng)
    path = np.concatenate([[0.0], np.cumsum(noise)])
    h_series = mfdfa(noise, Q, 2, SPEC).h[I2]
    h_path = mfdfa(path, Q, 2, SPEC).h[I2]
    assert h_path - h_series == pytest.approx(1.0, abs=0.12)


def test_scaling_range_enforces_the_segment_floor():
    spec = ScalingRange(policy="fixed", s_min=10, min_segments=10)
    scales = spec.candidate_scales(5000, order=2)
    assert scales.max() <= 500
    assert scales.min() >= 10


def test_scaling_range_rejects_a_series_that_is_too_short():
    with pytest.raises(ValueError):
        ScalingRange(s_min=10, min_segments=10).candidate_scales(50, order=2)


def test_mfdfa_rejects_non_finite_input():
    with pytest.raises(ValueError):
        mfdfa(np.array([1.0, np.nan] * 2000), Q, 2, SPEC)


def test_q_zero_is_the_geometric_mean_limit(rng):
    """F_0 must agree with the limit of F_q as q approaches zero."""
    from rvmf.mfdfa import fluctuation_functions

    x = fgn(2**12, 0.35, rng)
    scales = np.array([16, 32, 64, 128])
    f_zero = fluctuation_functions(x, [0.0], scales, order=2)[0]
    f_near = fluctuation_functions(x, [1e-6], scales, order=2)[0]
    assert np.allclose(f_zero, f_near, rtol=1e-4)


# ------------------------------------------------------------------------ ghe


def test_monofractal_spectrum_has_zero_width():
    h = np.full_like(Q, 0.3)
    assert spectrum_width(Q, h) == pytest.approx(0.0, abs=1e-9)


def test_anchored_fit_is_exact_on_a_line():
    big_h, lam = 0.27, 0.031
    h = big_h - lam * (Q - 2.0)
    fit = anchored_fit(Q, h)
    assert fit.H == pytest.approx(big_h)
    assert fit.lam == pytest.approx(lam)
    assert np.allclose(fit.xi, 0.0, atol=1e-12)
    assert fit.r2 == pytest.approx(1.0)


def test_anchored_fit_clamps_negative_slope():
    h = 0.3 + 0.02 * (Q - 2.0)  # increasing in q, so lambda is negative
    fit = anchored_fit(Q, h)
    assert fit.lam < 0
    assert fit.lam_plus == 0.0


def test_anchored_fit_requires_q_equals_two():
    grid = np.array([-3.0, -1.0, 1.0, 3.0])
    with pytest.raises(ValueError):
        anchored_fit(grid, np.zeros_like(grid))


def test_spectrum_requires_increasing_grid():
    with pytest.raises(ValueError):
        singularity_spectrum(Q[::-1], np.full_like(Q, 0.3))


# ----------------------------------------------------------------- surrogates


def test_shuffle_preserves_the_marginal_and_destroys_the_spectrum(rng):
    x = np.diff(lognormal_mmar(2**12 + 1, 0.30, 0.40, rng))
    s = shuffle_surrogate(x, rng)
    assert np.allclose(np.sort(x), np.sort(s))
    amp = lambda v: np.abs(np.fft.rfft(v))
    assert np.linalg.norm(amp(s) - amp(x)) / np.linalg.norm(amp(x)) > 0.2


def test_iaaft_preserves_both_the_marginal_and_the_spectrum(rng):
    x = np.diff(lognormal_mmar(2**12 + 1, 0.30, 0.40, rng))
    s = iaaft_surrogate(x, rng)
    assert np.allclose(np.sort(x), np.sort(s))
    amp = lambda v: np.abs(np.fft.rfft(v))
    assert np.linalg.norm(amp(s) - amp(x)) / np.linalg.norm(amp(x)) < 0.05


# ------------------------------------------------------------------- features


def test_no_feature_column_collapses_to_a_constant(rng):
    """The failure mode that pinned 32 of 44 columns to a floor."""
    x = fgn(5000, 0.25, rng)
    feats = extract_features(x, FeatureSpec(stride=10, standardise="none"))
    assert feats.shape[1] == 44
    assert (feats.std(axis=0) < 1e-6).sum() == 0


def test_per_order_standardisation_leaves_every_column_visible(rng):
    x = fgn(5000, 0.25, rng)
    feats = extract_features(x, FeatureSpec(stride=10, standardise="per_order"))
    sd = feats.std(axis=0)
    assert sd.min() > 0.05
    assert np.isfinite(feats).all()


def test_feature_names_and_groups_agree_with_the_matrix(rng):
    spec = FeatureSpec(stride=20)
    feats = extract_features(fgn(3000, 0.3, rng), spec)
    assert len(feature_names(spec)) == feats.shape[1] == spec.n_features
    assert feature_groups(spec).size == feats.shape[1]


def test_features_reject_a_series_shorter_than_the_largest_window(rng):
    with pytest.raises(ValueError):
        extract_features(fgn(100, 0.3, rng), FeatureSpec())
