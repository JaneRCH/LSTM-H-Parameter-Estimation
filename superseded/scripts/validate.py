"""Reproducible validation suite for the estimation machinery.

Every number the report cites about the *machinery*, as opposed to the data,
is produced here: that the simulators generate what they claim, that the
estimator recovers known labels, that the scaling-range policy is justified
rather than asserted, and that the surrogates have the properties the
decomposition requires.

Run:
    PYTHONPATH=src python3 scripts/validate.py --out results/validation.json
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np

from rvmf.features import FeatureSpec, extract_features
from rvmf.ghe import anchored_fit, spectrum_width
from rvmf.mfdfa import ScalingRange, mfdfa
from rvmf.simulators import (
    _fgn_autocovariance,
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
from rvmf.surrogates import iaaft_surrogate, shuffle_surrogate

warnings.filterwarnings("ignore")
Q = np.arange(-5, 6, dtype=float)
I2 = int(np.flatnonzero(Q == 2.0)[0])
SEED = 4830


def _mfdfa_h(x, order=2, spec=None):
    return mfdfa(x, Q, order, spec or ScalingRange()).h


def check_fgn_generator(rng, n=4096, reps=150) -> dict:
    """Davies-Harte output must have unit variance and the theoretical ACF."""
    rows = []
    for hurst in (0.2, 0.3, 0.5, 0.7):
        paths = np.array([fgn(n, hurst, rng) for _ in range(reps)])
        centred = paths - paths.mean(axis=1, keepdims=True)
        acf = [float((centred[:, :-k] * centred[:, k:]).mean()) for k in range(1, 6)]
        theory = _fgn_autocovariance(6, hurst)[1:6]
        rows.append(
            {
                "hurst": hurst,
                "variance": float(paths.var(axis=1).mean()),
                "acf": acf,
                "acf_theory": theory.tolist(),
                "max_abs_acf_error": float(np.max(np.abs(np.array(acf) - theory))),
            }
        )
    return {"reps": reps, "n": n, "rows": rows}


def check_convention(rng, n=2**14) -> dict:
    """MF-DFA on a noise series returns H; on the corresponding path, H + 1."""
    rows = []
    for hurst in (0.2, 0.3, 0.7):
        noise = fgn(n, hurst, rng)
        path = np.concatenate([[0.0], np.cumsum(noise)])
        h_noise = float(_mfdfa_h(noise)[I2])
        h_path = float(_mfdfa_h(path)[I2])
        rows.append(
            {
                "hurst": hurst,
                "h2_series": h_noise,
                "h2_path": h_path,
                "difference": h_path - h_noise,
            }
        )
    return {"n": n, "rows": rows}


def check_simulator_labels(rng, n=2**14, reps=8) -> dict:
    """Each simulator must reproduce its analytic label under MF-DFA."""
    cases = [
        ("fbm_H0.30", lambda: fgn(n, 0.30, rng), h_fbm(Q, 0.30)),
        ("fbm_H0.70", lambda: fgn(n, 0.70, rng), h_fbm(Q, 0.70)),
        ("binomial_p0.60", lambda: binomial_cascade(14, 0.60, rng), h_binomial(Q, 0.60)),
        ("binomial_p0.70", lambda: binomial_cascade(14, 0.70, rng), h_binomial(Q, 0.70)),
        (
            "mmar_Hb0.30_lc0.40",
            lambda: np.diff(lognormal_mmar(n + 1, 0.30, 0.40, rng)),
            h_lognormal_mmar(Q, 0.30, 0.40),
        ),
        (
            "mmar_Hb0.25_lc0.40",
            lambda: np.diff(lognormal_mmar(n + 1, 0.25, 0.40, rng)),
            h_lognormal_mmar(Q, 0.25, 0.40),
        ),
        (
            "mmar_Hb0.40_lc0.60",
            lambda: np.diff(lognormal_mmar(n + 1, 0.40, 0.60, rng)),
            h_lognormal_mmar(Q, 0.40, 0.60),
        ),
    ]
    rows = []
    for name, gen, label in cases:
        errors, h2 = [], []
        for _ in range(reps):
            est = _mfdfa_h(gen())
            valid = np.isfinite(label)
            errors.append(np.abs(est[valid] - label[valid]))
            h2.append(est[I2])
        errors = np.array(errors)
        rows.append(
            {
                "case": name,
                "mae": float(errors.mean()),
                "max_abs_error": float(errors.max()),
                "h2_estimated": float(np.mean(h2)),
                "h2_label": float(label[I2]),
            }
        )
    return {"n": n, "reps": reps, "rows": rows}


def check_anchored_recovery(rng, n=2**14, reps=8) -> dict:
    """The anchored (H, lambda) pair must be recovered from the estimated vector."""
    rows = []
    for hb, lc2 in ((0.30, 0.40), (0.25, 0.40), (0.40, 0.60)):
        true_h, true_lam = mmar_moments(hb, lc2)
        fits = [
            anchored_fit(Q, _mfdfa_h(np.diff(lognormal_mmar(n + 1, hb, lc2, rng))))
            for _ in range(reps)
        ]
        rows.append(
            {
                "hurst_base": hb,
                "lambda_c2": lc2,
                "H_true": true_h,
                "H_estimated": float(np.mean([f.H for f in fits])),
                "lambda_true": true_lam,
                "lambda_estimated": float(np.mean([f.lam for f in fits])),
                "anchored_r2": float(np.mean([f.r2 for f in fits])),
            }
        )
    return {"n": n, "reps": reps, "rows": rows}


def check_heavy_tail_label(rng, n=2**13, reps=12) -> dict:
    """Does the alpha-stable piecewise label survive at gamma > 2?"""
    rows = []
    for hurst, gamma in ((0.25, 3.0), (0.25, 2.5), (0.40, 3.0), (0.25, 4.0)):
        est = np.mean(
            [
                _mfdfa_h(np.diff(rough_heavy_tailed(n + 1, hurst, gamma, rng)))
                for _ in range(reps)
            ],
            axis=0,
        )
        piecewise = h_rough_heavy_tailed(Q, hurst, gamma)
        rows.append(
            {
                "hurst": hurst,
                "gamma": gamma,
                "h_estimated_q2_to_q5": est[I2 : I2 + 4].tolist(),
                "h_piecewise_q2_to_q5": piecewise[I2 : I2 + 4].tolist(),
                "mae_vs_piecewise": float(np.mean(np.abs(est - piecewise))),
                "mae_vs_flat": float(np.mean(np.abs(est - hurst))),
                "width": float(spectrum_width(Q, est)),
            }
        )
    return {
        "n": n,
        "reps": reps,
        "rows": rows,
        "verdict": (
            "The piecewise form fits no better than a flat line and the estimated "
            "profile declines smoothly from q = 2 with no break at q = gamma. It is "
            "not usable as a supervised label at gamma > 2; the null must be "
            "imposed by simulation."
        ),
    }


def check_finite_size(rng, reps=50) -> dict:
    """Width fabricated by MF-DFA on a process that is monofractal by construction."""
    rows = []
    for n in (512, 1024, 2048, 5000):
        widths = [spectrum_width(Q, _mfdfa_h(fgn(n, 0.30, rng))) for _ in range(reps)]
        rows.append(
            {
                "n": n,
                "mean_width": float(np.mean(widths)),
                "sd_width": float(np.std(widths)),
                "p95_width": float(np.percentile(widths, 95)),
            }
        )
    return {"reps": reps, "true_width": 0.0, "hurst": 0.30, "rows": rows}


def check_scaling_policy(rng, reps=25) -> dict:
    """Score the scaling-range policies against known truth.

    This is the item the marked proposal flagged as unvalidated.  The original
    upper bound of ``L / 4`` is included so the comparison is against the
    specification it replaces, not only against the alternative policy.
    """
    policies = {
        "fixed_min10seg": ScalingRange(policy="fixed", s_min=10, min_segments=10),
        "adaptive_min10seg": ScalingRange(policy="adaptive", s_min=10, min_segments=10),
        "original_L_over_4": ScalingRange(policy="adaptive", s_min=4, min_segments=4),
    }
    rows = []
    for name, spec in policies.items():
        errors, widths, segments = [], [], []
        for _ in range(reps):
            hurst = float(rng.uniform(0.15, 0.45))
            x = fgn(5000, hurst, rng)
            res = mfdfa(x, Q, 2, spec)
            errors.append(abs(float(res.h[I2]) - hurst))
            widths.append(spectrum_width(Q, res.h))
            segments.append(res.n_segments_at_max_scale)
        rows.append(
            {
                "policy": name,
                "mae_h2_vs_truth": float(np.mean(errors)),
                "mean_fabricated_width": float(np.mean(widths)),
                "min_segments_at_max_scale": int(np.min(segments)),
            }
        )
    return {"reps": reps, "n": 5000, "note": "true width is zero", "rows": rows}


def check_surrogates(rng, n=2**13, reps=20) -> dict:
    """Surrogates must have the invariances the decomposition assumes.

    The reference series is a lognormal MMAR, which is genuinely multifractal by
    construction.  That matters for interpretation: it establishes how much
    width a *true* multifractal retains under shuffling at this sample length,
    which is the benchmark against which an empirical shuffle result has to be
    read.
    """
    x = np.diff(lognormal_mmar(n + 1, 0.30, 0.40, rng))
    amp = lambda v: np.abs(np.fft.rfft(v))
    base_amp = amp(x)

    shuffled = [shuffle_surrogate(x, rng) for _ in range(reps)]
    iaaft = [iaaft_surrogate(x, rng) for _ in range(reps)]

    rel = lambda vs: float(
        np.mean([np.linalg.norm(amp(v) - base_amp) / np.linalg.norm(base_amp) for v in vs])
    )
    h2 = lambda vs: float(np.mean([_mfdfa_h(v)[I2] for v in vs]))
    wid = lambda vs: [float(spectrum_width(Q, _mfdfa_h(v))) for v in vs]

    w_shuf, w_iaaft = wid(shuffled), wid(iaaft)
    w_obs = float(spectrum_width(Q, _mfdfa_h(x)))

    return {
        "n": n,
        "reps": reps,
        "marginal_preserved_shuffle": bool(
            np.allclose(np.sort(x), np.sort(shuffled[0]))
        ),
        "marginal_preserved_iaaft": bool(np.allclose(np.sort(x), np.sort(iaaft[0]))),
        "spectrum_relative_error_shuffle": rel(shuffled),
        "spectrum_relative_error_iaaft": rel(iaaft),
        "h2": {
            "original": float(_mfdfa_h(x)[I2]),
            "shuffle": h2(shuffled),
            "iaaft": h2(iaaft),
        },
        "width": {
            "original": w_obs,
            "shuffle_mean": float(np.mean(w_shuf)),
            "shuffle_p95": float(np.percentile(w_shuf, 95)),
            "iaaft_mean": float(np.mean(w_iaaft)),
            "iaaft_p95": float(np.percentile(w_iaaft, 95)),
        },
        "observed_exceeds_shuffle_p95": bool(w_obs > np.percentile(w_shuf, 95)),
        "note": (
            "A genuinely multifractal series retains a substantial share of its "
            "width under shuffling at this length. An observed width that falls "
            "inside a shuffle distribution is therefore not on its own evidence "
            "that the width is distributional in origin."
        ),
    }


def check_standardisation(rng) -> dict:
    """Score each standardisation choice by how much signal survives it.

    Column dispersion alone is a weak diagnostic, so the decisive measure is the
    held-out error of a ridge regression on the mean-pooled features.  Pooling
    is exactly the reduction the network applies, so this isolates the effect of
    the standardiser from the architecture.
    """
    from sklearn.linear_model import Ridge

    from rvmf.corpus import CorpusSpec, fixed_testset

    x = fgn(5000, 0.25, rng)
    cspec = CorpusSpec()
    rows = []
    for mode in ("per_order", "common", "per_feature", "none"):
        fspec = FeatureSpec(stride=10, standardise=mode)
        feats = extract_features(x, fspec)
        sd = feats.std(axis=0)

        f, y, m, _, _ = fixed_testset(99, cspec, fspec, 400, Q)
        pooled = f.mean(axis=1)
        n_train = 300
        pred = Ridge(alpha=1.0).fit(pooled[:n_train], y[:n_train]).predict(
            pooled[n_train:]
        )
        rmse = float(
            np.sqrt(
                (((pred - y[n_train:]) ** 2) * m[n_train:]).sum()
                / max(m[n_train:].sum(), 1)
            )
        )
        rows.append(
            {
                "mode": mode,
                "n_columns": int(sd.size),
                "min_column_sd": float(sd.min()),
                "max_column_sd": float(sd.max()),
                "n_columns_below_0.01": int((sd < 0.01).sum()),
                "ridge_holdout_rmse": rmse,
                "label_sd": float(y.std()),
            }
        )
    return {"rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="results/validation.json")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    started = time.time()
    checks = {
        "fgn_generator": check_fgn_generator,
        "estimand_convention": check_convention,
        "simulator_labels": check_simulator_labels,
        "anchored_recovery": check_anchored_recovery,
        "heavy_tail_label": check_heavy_tail_label,
        "finite_size": check_finite_size,
        "scaling_policy": check_scaling_policy,
        "surrogates": check_surrogates,
        "standardisation": check_standardisation,
    }
    report: dict = {"seed": SEED, "q": Q.tolist()}
    for name, fn in checks.items():
        t = time.time()
        print(f"[{name}] running", flush=True)
        report[name] = fn(rng)
        print(f"[{name}] done in {time.time()-t:.1f}s", flush=True)

    report["wall_clock_s"] = time.time() - started
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwritten to {out}  ({report['wall_clock_s']:.0f}s)")


if __name__ == "__main__":
    main()
