"""Unit tests for the log S-fBM simulator and the OHLC observation model."""

from __future__ import annotations

import numpy as np
import pytest

from rvmf.gmm_lnm import default_lags, empirical_D, estimate, model_D
from rvmf.observation import ObservationParams, simulate_bars
from rvmf.rangeproxy import PROXIES, log_proxy, proxy_variance
from rvmf.seeds import MASTER_SEED, generator, seed_table
from rvmf.sfbm import (
    CUTOFF_RATIO,
    SfbmParams,
    autocovariance,
    embedding_defect,
    increment_variance,
    nu_squared,
    simulate_omega,
)


class TestSeeds:
    def test_master_seed_is_the_student_number(self):
        assert MASTER_SEED == 56233043

    def test_streams_are_independent(self):
        a = generator("corpus_train").standard_normal(1000)
        b = generator("corpus_test").standard_normal(1000)
        assert abs(float(np.corrcoef(a, b)[0, 1])) < 0.1

    def test_streams_are_reproducible(self):
        a = generator("nulls").standard_normal(50)
        b = generator("nulls").standard_normal(50)
        assert np.allclose(a, b)

    def test_unknown_stream_is_rejected(self):
        with pytest.raises(KeyError):
            generator("not_a_stream")

    def test_every_stream_has_a_distinct_seed(self):
        table = seed_table()
        assert len(set(table.values())) == len(table)


class TestCovariance:
    def test_intermittency_relation(self):
        assert nu_squared(0.13, 0.05) == pytest.approx(0.05 / (0.13 * 0.74))

    def test_multifractal_limit(self):
        """As H falls to zero the covariance approaches lambda^2 log(T/tau)."""
        delta, lam, T = 1 / 48, 0.05, 250.0
        tau = np.arange(1, 400) * delta
        target = lam * np.log(T / tau)
        gaps = [
            float(np.max(np.abs(autocovariance(400, delta, SfbmParams(H, lam, T))[1:] - target) / target))
            for H in (0.05, 0.01, 0.001)
        ]
        assert gaps[0] > gaps[1] > gaps[2]
        assert gaps[2] < 0.02

    def test_covariance_vanishes_beyond_the_correlation_scale(self):
        p = SfbmParams(H=0.13, lambda_sq=0.05, T=2.0)
        cov = autocovariance(400, 1 / 48, p)
        assert np.all(cov[int(2.0 * 48) :] == 0.0)

    @pytest.mark.parametrize("H", [0.0, 0.05, 0.13, 0.30])
    def test_simulation_recovers_the_target_covariance(self, H):
        p = SfbmParams(H=H, lambda_sq=0.05, T=250.0)
        n, delta = 2048, 1 / 48
        rng = generator("corpus_val")
        paths = np.array([simulate_omega(n, delta, p, rng) for _ in range(250)])
        target = autocovariance(n, delta, p)
        for k in (0, 1, 10, 100):
            emp = float(np.mean(paths[:, : n - k] * paths[:, k:]))
            assert abs(emp - target[k]) < 0.08 * target[0]

    def test_embedding_defect_is_small(self):
        p = SfbmParams(H=0.13, lambda_sq=0.05, T=25000.0)
        assert embedding_defect(100000, 1 / 48, p) < 0.05


class TestBarConvention:
    """The simulator must reproduce the convention of the Yahoo files."""

    @staticmethod
    def _bars():
        return simulate_bars(
            1500,
            SfbmParams(H=0.13, lambda_sq=0.05, T=5000.0),
            ObservationParams(),
            generator("corpus_val"),
        )

    def test_open_equals_previous_close(self):
        b = self._bars()
        assert np.allclose(b["open"][1:], b["close"][:-1])

    def test_high_and_low_bracket_open_and_close(self):
        b = self._bars()
        assert np.all(b["high"] >= np.maximum(b["open"], b["close"]) - 1e-12)
        assert np.all(b["low"] <= np.minimum(b["open"], b["close"]) + 1e-12)

    def test_range_is_usually_wider_than_open_to_close(self):
        """In the JSE files the high equals max(open, close) on 19 to 27 percent
        of days. A simulator that produced a degenerate range would not."""
        b = self._bars()
        share = float(np.mean(np.isclose(b["high"], np.maximum(b["open"], b["close"]))))
        assert 0.05 < share < 0.45

    def test_daily_return_is_close_minus_open(self):
        b = self._bars()
        assert np.allclose(b["ret"], b["close"] - b["open"])


class TestProxies:
    def test_parkinson_and_garman_klass_are_less_noisy_than_squared_returns(self):
        b = simulate_bars(
            6000,
            SfbmParams(H=0.13, lambda_sq=0.05, T=20000.0),
            ObservationParams(),
            generator("corpus_val"),
        )
        truth = np.log(b["integrated_variance"])
        noise = {k: float(np.var(log_proxy(b, k) - truth)) for k in PROXIES}
        assert noise["garman_klass"] < noise["rv"] / 5.0
        assert noise["parkinson"] < noise["rv"] / 5.0

    def test_unknown_proxy_is_rejected(self):
        with pytest.raises(ValueError):
            proxy_variance({"open": np.zeros(2), "high": np.zeros(2),
                            "low": np.zeros(2), "close": np.zeros(2)}, "nope")


class TestGmmMoments:
    def test_zero_lag_difference_vanishes(self):
        for H in (0.0, 0.1, 0.3):
            assert model_D(np.array([0]), H, 0.05, 0.0)[0] == pytest.approx(0.0, abs=1e-12)

    def test_moments_decrease_with_lag(self):
        d = model_D(default_lags(256), 0.13, 0.05, 0.0)
        assert np.all(np.diff(d) < 0)

    def test_noise_enters_as_a_constant_shift(self):
        lags = default_lags(64)
        a = model_D(lags, 0.13, 0.05, 0.0)
        b = model_D(lags, 0.13, 0.05, 0.4)
        assert np.allclose(a - b, 0.4)

    def test_analytic_moments_match_the_simulated_latent_process(self):
        H, lam, m, days = 0.13, 0.05, 48, 3000
        p = SfbmParams(H=H, lambda_sq=lam, T=30000.0)
        rng = generator("corpus_val")
        lags = default_lags(128)
        emp = np.mean(
            [
                empirical_D(simulate_omega(days * m, 1 / m, p, rng).reshape(days, m).mean(axis=1), lags)
                for _ in range(40)
            ],
            axis=0,
        )
        model = model_D(lags, H, lam, 0.0)
        assert np.max(np.abs(emp - model)) < 0.12 * abs(model[-1])

    def test_estimator_recovers_a_known_latent_process(self):
        """End to end on the latent daily average, where noise is absent."""
        H, lam, m, days = 0.15, 0.06, 48, 5100
        p = SfbmParams(H=H, lambda_sq=lam, T=30000.0)
        rng = generator("identification")
        errs = []
        for _ in range(8):
            w = simulate_omega(days * m, 1 / m, p, rng).reshape(days, m).mean(axis=1)
            errs.append(estimate(w).H - H)
        assert abs(float(np.mean(errs))) < 0.06


# ----------------------------------------------------------------------------
# Regression guards for the small-scale cutoff.
#
# An earlier version evaluated the covariance at tau_k = max(k * delta, delta),
# which put the cutoff AT the grid step and so gave C(delta) == C(0) exactly.
# The lag-one increment of the model then had zero variance: the process was
# degenerate at the finest resolved scale, and the variation the simulator
# produced there came entirely from clipping negative circulant eigenvalues.
# The suite passed throughout, because nothing compared the simulated structure
# function to its analytic value. These tests close that gap.
# ----------------------------------------------------------------------------

_CUTOFF_GRID = [(0.00, 0.02), (0.05, 0.02), (0.10, 0.05), (0.20, 0.05),
                (0.30, 0.10), (0.45, 0.10)]


@pytest.mark.parametrize("H,lambda_sq", _CUTOFF_GRID)
def test_cutoff_sits_below_the_grid(H, lambda_sq):
    """C(0) must exceed C(delta) strictly, or the lag-one increment vanishes."""
    p = SfbmParams(H=H, lambda_sq=lambda_sq, T=20400.0)
    cov = autocovariance(3, 1.0, p)
    assert cov[0] > cov[1], f"degenerate at the grid scale: C(0)={cov[0]}, C(1)={cov[1]}"
    # The lag-one increment variance is 2(C(0) - C(delta)) and must be positive.
    assert 2.0 * (cov[0] - cov[1]) > 0.0


def test_cutoff_ratio_is_strictly_below_one():
    assert 0.0 < CUTOFF_RATIO < 1.0


def test_multifractal_limit_lag_one_increment():
    """At H = 0 the cutoff choice fixes the lag-one increment variance at 2 lambda^2."""
    lambda_sq = 0.037
    p = SfbmParams(H=0.0, lambda_sq=lambda_sq, T=20400.0)
    got = float(increment_variance(np.array([1.0]), 1.0, p)[0])
    assert got == pytest.approx(2.0 * lambda_sq, rel=1e-12)


@pytest.mark.parametrize("H,lambda_sq", _CUTOFF_GRID)
def test_increment_variance_matches_covariance(H, lambda_sq):
    """The analytic structure function must agree with 2(C(0) - C(tau))."""
    p = SfbmParams(H=H, lambda_sq=lambda_sq, T=20400.0)
    lags = np.array([1.0, 2.0, 8.0, 64.0, 512.0])
    cov = autocovariance(int(lags.max()) + 1, 1.0, p)
    from_cov = 2.0 * (cov[0] - cov[lags.astype(int)])
    assert increment_variance(lags, 1.0, p) == pytest.approx(from_cov, rel=1e-10)


@pytest.mark.parametrize("H,lambda_sq", _CUTOFF_GRID)
def test_simulated_structure_function_matches_analytic(H, lambda_sq):
    """The simulator must reproduce the analytic structure function at every scale.

    This is the check the old suite lacked. The marginal variance is NOT a valid
    target at this sample length: the correlation scale T exceeds the window, so
    the long-wavelength component is removed with the sample mean and the sample
    variance understates the marginal variance by construction.
    """
    n, delta, reps = 2048, 1.0, 24
    p = SfbmParams(H=H, lambda_sq=lambda_sq, T=4.0 * n * delta)
    rng = np.random.default_rng(MASTER_SEED)
    paths = np.stack([simulate_omega(n, delta, p, rng) for _ in range(reps)])
    lags = np.array([1, 2, 4, 16, 64, 256])
    for lag in lags:
        d = paths[:, lag:] - paths[:, :-lag]
        got = float((d ** 2).mean())
        want = float(increment_variance(np.array([lag * delta]), delta, p)[0])
        assert got == pytest.approx(want, rel=0.10), (
            f"H={H} lag={lag}: simulated {got:.5f} against analytic {want:.5f}"
        )


@pytest.mark.parametrize("H,lambda_sq", _CUTOFF_GRID)
def test_embedding_is_essentially_exact(H, lambda_sq):
    """With the cutoff below the grid the circulant extension is embeddable.

    The old cutoff lowered C(0) and so cost positive definiteness: the clipped
    mass reached 1.3 percent. Raising the diagonal restores it.
    """
    p = SfbmParams(H=H, lambda_sq=lambda_sq, T=20400.0)
    assert embedding_defect(5100, 1.0, p) < 1e-5


@pytest.mark.parametrize("H,lambda_sq", [(0.0, 0.02), (0.20, 0.05), (0.45, 0.10)])
def test_simulated_increments_are_gaussian(H, lambda_sq):
    """omega is Gaussian, so its increments must have kurtosis near three."""
    n, reps = 2048, 24
    p = SfbmParams(H=H, lambda_sq=lambda_sq, T=4.0 * n)
    rng = np.random.default_rng(MASTER_SEED)
    paths = np.stack([simulate_omega(n, 1.0, p, rng) for _ in range(reps)])
    d = (paths[:, 1:] - paths[:, :-1]).ravel()
    kurtosis = float(((d - d.mean()) ** 4).mean() / d.var() ** 2)
    assert kurtosis == pytest.approx(3.0, abs=0.15)


# ----------------------------------------------------------------------------
# Regression guard for the contaminated-print screen.
#
# The interim report described this screen before it existed in the pipeline.
# One uncaught print in J203 (a close of 179.80 against a previous close of
# 99,324.53, reversed the next day) raised the kurtosis of the log Garman-Klass
# increments from 3.2 to 15.2 and drove the learned estimator below H = 0, which
# is outside the support of the model. These tests keep the screen honest.
# ----------------------------------------------------------------------------


def _bar_frame(closes):
    import pandas as pd
    closes = np.asarray(closes, dtype=float)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    return pd.DataFrame(
        {"open": opens, "high": np.maximum(opens, closes) * 1.002,
         "low": np.minimum(opens, closes) * 0.998, "close": closes},
        index=pd.date_range("2020-01-01", periods=closes.size, freq="B"),
    )


def test_screen_removes_a_reversing_print():
    from rvmf.data import screen_contaminated_prints
    closes = [100.0] * 10
    closes[5] = 0.18          # a vendor print three orders of magnitude out
    frame = _bar_frame(closes)
    out, dropped = screen_contaminated_prints(frame)
    assert dropped == 2, "both the contaminated bar and the recovery bar go"
    assert 0.18 not in out["close"].to_numpy()
    assert 0.18 not in out["open"].to_numpy(), "the recovery bar opens at the bad level"


def test_screen_keeps_a_genuine_crash():
    """A large move that does not reverse is a market move, not a print."""
    from rvmf.data import screen_contaminated_prints
    closes = [100.0] * 5 + [70.0] * 5          # a 30 percent fall that persists
    out, dropped = screen_contaminated_prints(_bar_frame(closes))
    assert dropped == 0
    assert len(out) == len(closes)


def test_screen_keeps_a_large_round_trip_that_does_not_restore():
    """Down then up, but not back to where it started, is two real moves."""
    from rvmf.data import screen_contaminated_prints
    closes = [100.0, 100.0, 60.0, 85.0, 85.0, 85.0]
    out, dropped = screen_contaminated_prints(_bar_frame(closes))
    assert dropped == 0


def test_screen_is_a_no_op_on_clean_data():
    from rvmf.data import screen_contaminated_prints
    rng = np.random.default_rng(MASTER_SEED)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 500)))
    out, dropped = screen_contaminated_prints(_bar_frame(closes))
    assert dropped == 0
    assert len(out) == 500


# ---------------------------------------------------------------------------
# Recording convention. The simulator records every open as the previous close
# and lets the range span it. Four of the seven index files satisfy that
# convention on better than 99 per cent of days and three do not, so a file's
# convention has to be measured and imposed rather than assumed.


def _gapped_frame(closes: list[float], gaps: list[float]) -> "pd.DataFrame":
    """Bars whose open is the previous close moved by a stated overnight gap."""
    import pandas as pd
    closes = np.asarray(closes, dtype=float)
    opens = np.concatenate([[closes[0]], closes[:-1] * (1.0 + np.asarray(gaps, float))])
    return pd.DataFrame(
        {"open": opens,
         "high": np.maximum(opens, closes) * 1.002,
         "low": np.minimum(opens, closes) * 0.998,
         "close": closes},
        index=pd.date_range("2020-01-01", periods=closes.size, freq="B"),
    )


def test_stale_open_share_detects_a_gapped_file():
    from rvmf.data import stale_open_share
    n = 100
    closes = list(100.0 + np.arange(n) * 0.01)
    gaps = [0.0] * (n - 1)
    for i in range(0, n - 1, 4):               # a real open on one day in four
        gaps[i] = 0.01
    share = stale_open_share(_gapped_frame(closes, gaps))
    assert 0.70 < share < 0.80, share


def test_stale_open_share_is_one_on_a_stale_file():
    from rvmf.data import stale_open_share
    closes = list(100.0 + np.arange(50) * 0.01)
    assert stale_open_share(_gapped_frame(closes, [0.0] * 49)) == 1.0


def test_impose_stale_open_matches_the_simulator_convention():
    """After the transformation every open is the previous close exactly."""
    from rvmf.data import impose_stale_open, stale_open_share
    rng = np.random.default_rng(MASTER_SEED)
    closes = list(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 200))))
    gaps = list(rng.normal(0, 0.004, 199))
    out = impose_stale_open(_gapped_frame(closes, gaps))
    assert stale_open_share(out) == 1.0
    assert len(out) == 199, "the first bar has no previous close"


def test_impose_stale_open_keeps_ohlc_integrity():
    """The widened range must still contain the open and the close."""
    from rvmf.data import impose_stale_open
    rng = np.random.default_rng(MASTER_SEED)
    closes = list(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 200))))
    out = impose_stale_open(_gapped_frame(closes, list(rng.normal(0, 0.01, 199))))
    o, h, l, c = (out[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    assert np.all(l <= np.minimum(o, c) + 1e-12)
    assert np.all(np.maximum(o, c) <= h + 1e-12)


def test_impose_stale_open_is_near_identity_on_a_stale_file():
    """On a file that already satisfies the convention nothing moves."""
    from rvmf.data import impose_stale_open
    rng = np.random.default_rng(MASTER_SEED)
    closes = list(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 200))))
    frame = _gapped_frame(closes, [0.0] * 199)
    out = impose_stale_open(frame)
    for key in ("open", "high", "low", "close"):
        assert np.allclose(out[key].to_numpy(float),
                           frame[key].to_numpy(float)[1:], rtol=1e-12)


def test_simulated_bars_satisfy_the_convention_the_files_are_held_to():
    """The guard that ties the two halves together.

    Whatever is imposed on the data must be what the simulator produces, or the
    estimator is trained on one convention and applied to another.
    """
    from rvmf.data import stale_open_share
    import pandas as pd
    rng = generator("tests")
    bars = simulate_bars(400, SfbmParams(H=0.10, lambda_sq=0.03, T=4000.0,
                                         sigma_bar=0.01),
                         ObservationParams(), rng)
    frame = pd.DataFrame({k: np.exp(bars[k]) for k in
                          ("open", "high", "low", "close")})
    assert stale_open_share(frame) > 0.999


# ---------------------------------------------------------------------------
# Pooling rule. The moment estimator and the learned estimator are compared
# against one threshold, so their cell errors have to be pooled the same way.


def test_pooled_rmse_is_the_rmse_over_equally_sized_cells():
    """Pooling cell errors must equal the error over all their replicates."""
    from rvmf.diagnostics import pooled_rmse
    rng = np.random.default_rng(MASTER_SEED)
    cells = [rng.normal(0.0, s, 500) for s in (0.02, 0.05, 0.09)]
    per_cell = [np.sqrt(np.mean(c ** 2)) for c in cells]
    assert np.isclose(pooled_rmse(per_cell),
                      np.sqrt(np.mean(np.concatenate(cells) ** 2)))


def test_pooled_rmse_exceeds_the_mean_when_cells_differ():
    """The guard on the defect this replaced.

    The two identification scripts once pooled by different rules, the mean in
    one and the root mean square in the other, so the figures they produced were
    not comparable. The two coincide only when every cell is equal.
    """
    from rvmf.diagnostics import pooled_rmse
    uneven = [0.0378, 0.0267, 0.0187]
    assert pooled_rmse(uneven) > float(np.mean(uneven))
    assert np.isclose(pooled_rmse([0.04] * 3), 0.04)


def test_both_identification_scripts_use_the_shared_pooling_rule():
    """A grep-level guard: neither script may pool inline again."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "scripts"
    for name in ("identification_check.py", "identification_net.py"):
        source = (root / name).read_text()
        assert "pooled_rmse" in source, name
        assert "np.mean([r[" not in source, f"{name} pools inline again"
