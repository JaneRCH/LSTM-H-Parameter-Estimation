"""JSE empirical analysis.

Loads the index data, applies the cleaning protocol and the vendor rule,
constructs both volatility proxies, estimates h(q) by MF-DFA and by the trained
network, derives the anchored (H, lambda) pair, and tests the observed spectrum
width against the monofractal, rough heavy-tailed and distributional nulls.

Run:
    PYTHONPATH=src python3 scripts/empirical.py --data-dir /path/to/data
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import warnings

import numpy as np
import torch

warnings.filterwarnings('ignore')

from rvmf.corpus import DEFAULT_Q
from rvmf.data import clean_ohlc, load_ohlc, reconcile
from rvmf.diagnostics import descriptives, hill_estimator, hill_plateau, stationarity
from rvmf.features import FeatureSpec, extract_features
from rvmf.ghe import anchored_fit, singularity_spectrum, spectrum_width
from rvmf.mfdfa import ScalingRange, mfdfa
from rvmf.model import HurstLSTM, ModelConfig, predict_numpy
from rvmf.nulls import monte_carlo_test
from rvmf.proxies import Estimand, build_estimand
from rvmf.surrogates import iaaft_surrogate, shuffle_surrogate

Q = DEFAULT_Q
I2 = int(np.flatnonzero(Q == 2.0)[0])
SEED = 4834


def load_model(path: str):
    ckpt = torch.load(path, weights_only=False)
    cfg = ModelConfig(**ckpt["config"])
    model = HurstLSTM(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def lstm_h(model, x: np.ndarray, fspec: FeatureSpec) -> np.ndarray:
    feats = extract_features(np.asarray(x, dtype=float), fspec)[None, ...]
    return predict_numpy(model, feats)[0]


def shift_diagnostic(x, fspec, ref_feats) -> float:
    """Share of the empirical feature values outside the training central band.

    A learned estimator applied outside the support it was trained on is
    extrapolating, and this reports how far outside.
    """
    feats = extract_features(np.asarray(x, dtype=float), fspec)
    lo = np.percentile(ref_feats, 1, axis=(0, 1))
    hi = np.percentile(ref_feats, 99, axis=(0, 1))
    outside = (feats < lo) | (feats > hi)
    return float(outside.mean())


def rolling_mean_transform(window: int):
    """Smooth a null replicate with the same window the proxy carries.

    A range estimator computed on a rolling window of ``n`` days induces
    autocorrelation over ``n`` lags by construction. Passing each null replicate
    through the identical window puts that induced dependence into the null and
    the observed statistic alike, so it cancels in the comparison rather than
    being attributed to the market.
    """

    def apply(v: np.ndarray) -> np.ndarray:
        c = np.cumsum(np.concatenate([[0.0], np.asarray(v, dtype=float)]))
        return (c[window:] - c[:-window]) / float(window)

    return apply


def analyse_series(
    name, x, model, fspec, spec, rng, n_null, n_surr, ref_feats, transform=None
):
    """Full estimation and testing for one estimand."""
    x = np.asarray(x, dtype=float)
    out = {"name": name, "n": int(x.size)}

    out["descriptives"] = descriptives(x)
    out["stationarity"] = stationarity(x)
    gamma = hill_estimator(np.diff(x))
    out["tail"] = {"gamma": gamma, "plateau": hill_plateau(np.diff(x))}

    res = mfdfa(x, Q, 2, spec)
    h_mf = res.h
    out["scaling_diagnostics"] = res.summary()
    out["mfdfa"] = {
        "h": h_mf.tolist(),
        "stderr": res.stderr.tolist(),
        "width": spectrum_width(Q, h_mf),
        "asymmetry": singularity_spectrum(Q, h_mf).asymmetry,
    }
    fit = anchored_fit(Q, h_mf)
    out["mfdfa"]["anchored"] = {
        "H": fit.H, "lambda": fit.lam, "lambda_plus": fit.lam_plus, "r2": fit.r2
    }

    h_ls = lstm_h(model, x, fspec)
    fit_ls = anchored_fit(Q, h_ls)
    out["lstm"] = {
        "h": h_ls.tolist(),
        "width": spectrum_width(Q, h_ls),
        "anchored": {
            "H": fit_ls.H, "lambda": fit_ls.lam,
            "lambda_plus": fit_ls.lam_plus, "r2": fit_ls.r2,
        },
        "feature_shift_outside_training_band": shift_diagnostic(x, fspec, ref_feats),
    }

    # Surrogate decomposition.
    w_obs = out["mfdfa"]["width"]
    shuf = [spectrum_width(Q, mfdfa(shuffle_surrogate(x, rng), Q, 2, spec).h)
            for _ in range(n_surr)]
    iaaft = [spectrum_width(Q, mfdfa(iaaft_surrogate(x, rng), Q, 2, spec).h)
             for _ in range(n_surr)]
    out["surrogates"] = {
        "n": n_surr,
        "observed_width": w_obs,
        "shuffle_mean": float(np.mean(shuf)),
        "shuffle_p95": float(np.percentile(shuf, 95)),
        "iaaft_mean": float(np.mean(iaaft)),
        "iaaft_p95": float(np.percentile(iaaft, 95)),
        "delta_pdf": float(np.mean(shuf)),
        "delta_nl": float(w_obs - np.mean(iaaft)),
        "observed_exceeds_shuffle_p95": bool(w_obs > np.percentile(shuf, 95)),
        "observed_exceeds_iaaft_p95": bool(w_obs > np.percentile(iaaft, 95)),
    }

    # Monte Carlo nulls, under MF-DFA.
    tests = {}
    for kind in ("monofractal", "heavy_tailed", "shuffle"):
        kwargs = {"tail_index": gamma} if kind == "heavy_tailed" else {}
        t0 = time.time()
        r = monte_carlo_test(
            x, kind, n_replicates=n_null, rng=rng, q=Q, order=2,
            scaling_range=spec, transform=transform, **kwargs,
        )
        tests[kind] = r.summary()
        tests[kind]["seconds"] = round(time.time() - t0, 1)
        # Keep the raw replicate widths so the report can plot the null
        # distribution itself rather than a sketch of its moments.
        tests[kind]["null_widths"] = [float(w) for w in r.null_widths]
        print(f"    null {kind:14s} p={r.p_value:.4f} "
              f"(obs {r.observed:.3f} vs null mean "
              f"{np.mean(r.null_widths):.3f})", flush=True)
    out["nulls"] = tests
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--checkpoint", default="results/lstm_best.pt")
    ap.add_argument("--out", default="results/empirical.json")
    ap.add_argument("--n-null", type=int, default=200)
    ap.add_argument("--n-surrogates", type=int, default=100)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument(
        "--primary",
        default="J200",
        choices=("J200", "J203", "J210", "J213"),
        help="index the primary analysis and the null tests run on. The "
        "cross-index sweep covers the remaining codes.",
    )
    args = ap.parse_args()

    torch.set_num_threads(2)
    d = Path(args.data_dir)
    rng = np.random.default_rng(SEED)
    fspec = FeatureSpec(stride=args.stride)
    spec = ScalingRange()  # the adopted policy
    model = load_model(args.checkpoint)

    # Reference feature distribution, for the shift diagnostic.
    from rvmf.corpus import CorpusSpec, fixed_testset
    ref_feats, *_ = fixed_testset(4836, CorpusSpec(), fspec, 128, Q)

    report = {"q": Q.tolist(), "seed": SEED, "policy": spec.policy,
              "primary_index": args.primary,
              "n_null": args.n_null, "n_surrogates": args.n_surrogates}

    # ---- Load, clean, reconcile ------------------------------------------
    frames, cleaning = {}, {}
    for code in ("J200", "J203", "J210", "J213"):
        raw = load_ohlc(d / f"OHLC_historical_data_{code}.csv")
        frames[code], rep = clean_ohlc(raw)
        cleaning[code] = rep.to_dict()
        cleaning[code]["start"] = str(frames[code].index.min().date())
        cleaning[code]["end"] = str(frames[code].index.max().date())
    report["cleaning"] = cleaning

    cross = d / "Jane_Chipfakacha_20260618235987.xlsx"
    if cross.exists():
        sec, _ = clean_ohlc(load_ohlc(cross, sheet="Price Data"))
        frames["J200"], rec = reconcile(frames["J200"], sec)
        report["reconciliation"] = rec.to_dict()
        if not rec.flagged.empty:
            report["reconciliation"]["flagged_dates"] = [
                {"date": str(i.date()), **{k: float(v) for k, v in r.items()
                                           if np.isfinite(v)}}
                for i, r in rec.flagged.iterrows()
            ]

    # ---- Primary analysis: the chosen index, both proxies ----------------
    report["primary"] = {}
    for proxy in ("rv", "yz"):
        est = Estimand(proxy=proxy, log=True, convention="series")
        x = build_estimand(frames[args.primary], est)

        # A rolling range estimator smooths the series over its window, so no
        # scale below the window may be read, and the null replicates must
        # carry the same smoothing. Both safeguards are applied here and
        # nowhere else, because only this proxy needs them.
        if proxy == "yz":
            proxy_spec = ScalingRange(
                policy=spec.policy,
                s_min=est.yz_window + 1,
                min_segments=spec.min_segments,
            )
            transform = rolling_mean_transform(est.yz_window)
        else:
            proxy_spec, transform = spec, None

        print(f"[{args.primary}/{est.label()}] n={len(x)} s_min={proxy_spec.s_min}",
              flush=True)
        report["primary"][est.label()] = analyse_series(
            f"{args.primary} {est.label()}", x.to_numpy(), model, fspec, proxy_spec, rng,
            args.n_null, args.n_surrogates, ref_feats, transform=transform,
        )

    # ---- Cross-index sweep, classical and learned, primary proxy ---------
    sweep = {}
    for code in [c for c in ("J200", "J203", "J210", "J213") if c != args.primary]:
        est = Estimand(proxy="rv", log=True, convention="series")
        x = build_estimand(frames[code], est).to_numpy()
        h_mf = mfdfa(x, Q, 2, spec).h
        h_ls = lstm_h(model, x, fspec)
        f_mf, f_ls = anchored_fit(Q, h_mf), anchored_fit(Q, h_ls)
        sweep[code] = {
            "n": int(x.size),
            "gamma": hill_estimator(np.diff(x)),
            "mfdfa": {"H": f_mf.H, "lambda": f_mf.lam,
                      "width": spectrum_width(Q, h_mf), "h": h_mf.tolist()},
            "lstm": {"H": f_ls.H, "lambda": f_ls.lam,
                     "width": spectrum_width(Q, h_ls), "h": h_ls.tolist()},
        }
        print(f"[{code}] MF-DFA H={f_mf.H:.3f} lam={f_mf.lam:.4f} | "
              f"LSTM H={f_ls.H:.3f} lam={f_ls.lam:.4f}", flush=True)
    report["cross_index"] = sweep

    # ---- Detrending-order sensitivity, policy held fixed -----------------
    est = Estimand(proxy="rv", log=True, convention="series")
    x = build_estimand(frames[args.primary], est).to_numpy()
    orders = {}
    for order in (1, 2, 3):
        r = mfdfa(x, Q, order, spec)
        fit = anchored_fit(Q, r.h)
        orders[order] = {
            "H": fit.H, "lambda": fit.lam,
            "width": spectrum_width(Q, r.h), **r.summary(),
        }
    report["detrending_sensitivity"] = orders

    Path(args.out).write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
