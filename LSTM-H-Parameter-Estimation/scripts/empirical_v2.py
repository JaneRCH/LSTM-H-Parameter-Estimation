"""Apply the estimators to the JSE indices and attach simulation-based intervals.

For each index the script reports ``(H, lambda^2)`` from the learned estimator
and from GMM, a parametric bootstrap interval for each, and two tests of
position on the rough-to-multifractal continuum: ``H = 0`` against ``H > 0``,
and ``H = 0.1``, the value the rough volatility literature reports for indices.

Intervals come from simulation rather than from an asymptotic formula.  At the
estimated parameters the whole observation model is re-simulated with nuisance
parameters drawn from the prior, the estimator is re-applied, and the quantiles
of the resulting error distribution give the interval.  This is the procedure of
:cite:`Boros2024` and it is the only honest option here, because the sampling
distribution of either estimator on daily range proxies has no closed form.

A prior predictive check is reported alongside: the share of the index's own
summary features that fall outside the central 98 percent band of the training
corpus.  A learned estimator applied outside the support it was trained on is
extrapolating, and the interim report's network was doing exactly that.

Run:
    PYTHONPATH=src python3 scripts/empirical_v2.py --replicates 400
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

from rvmf.corpus_v2 import PriorSpec, draw_parameters
from rvmf.data import (clean_ohlc, impose_stale_open, load_ohlc,
                       stale_open_share)
from rvmf.features_v2 import FeatureSpecV2, extract
from rvmf.gmm_lnm import estimate as gmm_estimate
from rvmf.model_v2 import ModelConfigV2, ThetaNet, predict_theta
from rvmf.observation import ObservationParams, simulate_bars
from rvmf.rangeproxy import log_proxy
from rvmf.seeds import generator
from rvmf.sfbm import SfbmParams

CODES = ("J200", "J203", "J210", "J213")

#: Stand-in for lambda^2 = 0, which the model's support excludes. At this value
#: the generalised Hurst function is flat to within Monte Carlo error; see the
#: correspondence established by scripts/apparent_hq.py.
LAMBDA_NULL = 1e-4


def _load_net(path: Path) -> ThetaNet:
    ckpt = torch.load(path, weights_only=False)
    model = ThetaNet(ModelConfigV2(**ckpt["config"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def _bars_from_frame(frame) -> dict[str, np.ndarray]:
    return {c: np.log(frame[c].to_numpy(dtype=float)) for c in ("open", "high", "low", "close")}


def _simulate_at(
    H: float, lam: float, length: int, prior: PriorSpec, spec: FeatureSpecV2,
    rng: np.random.Generator, net: ThetaNet, run_gmm: bool,
) -> tuple[float, float, float, float]:
    """One replicate at fixed targets with nuisance parameters redrawn.

    Returns the learned ``(H, lambda^2)`` and the GMM ``(H, lambda^2)``, the
    latter as NaN when ``run_gmm`` is false.
    """
    _, obs = draw_parameters(length, prior, rng)
    sf = SfbmParams(
        H=H, lambda_sq=lam,
        T=float(np.exp(rng.uniform(*np.log(prior.T_over_L)))) * length,
        sigma_bar=float(rng.uniform(*prior.sigma_bar)),
    )
    bars = simulate_bars(length, sf, obs, rng)
    seq, summ = extract(bars, spec)
    theta = predict_theta(net, seq[None, ...], summ[None, ...])[0]
    g = gmm_estimate(log_proxy(bars, spec.primary)) if run_gmm else None
    return (float(theta[0]), float(theta[1]),
            float(g.H) if g else np.nan,
            float(g.lambda_sq) if g else np.nan)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--net", default="results_v2/net/theta_full.pt")
    ap.add_argument("--train-corpus", default="results_v2/corpus/train_L5100_n40000.npz")
    ap.add_argument("--replicates", type=int, default=400)
    ap.add_argument("--gmm-replicates", type=int, default=150)
    ap.add_argument("--out", default="results_v2/empirical.json")
    ap.add_argument("--codes", nargs="*", default=list(CODES),
                    help="indices to run. Used to apply a length-matched network "
                         "to the subset of indices whose length it matches.")
    ap.add_argument("--length-label", type=int, default=5100,
                    help="the length the network was trained at, recorded in the "
                         "report and used for the length-match flag.")
    args = ap.parse_args()

    torch.set_num_threads(2)
    spec, prior = FeatureSpecV2(), PriorSpec()
    net = _load_net(Path(args.net))
    rng = generator("bootstrap")
    t0 = time.time()

    with np.load(args.train_corpus) as z:
        train_summary = z["summary"]
    lo = np.percentile(train_summary, 1, axis=0)
    hi = np.percentile(train_summary, 99, axis=0)

    report: dict = {"net": args.net, "replicates": args.replicates, "indices": {}}
    d = Path(args.data_dir)

    for code in args.codes:
        frame, _ = clean_ohlc(load_ohlc(d / f"OHLC_historical_data_{code}.csv"))
        frame = frame.dropna(subset=["open", "high", "low", "close"])
        # One recording convention for every file, matching the one the
        # simulator produces. Four of the seven already satisfy it on better
        # than 99 per cent of days and move by at most 0.004 in H; the other
        # three record a real open and would otherwise be read by an estimator
        # trained on bars built the other way.
        convention_before = stale_open_share(frame)
        frame = impose_stale_open(frame)
        bars = _bars_from_frame(frame)
        length = int(frame.shape[0])
        seq, summ = extract(bars, spec)

        theta = predict_theta(net, seq[None, ...], summ[None, ...])[0]
        H_net, lam_net = float(theta[0]), float(theta[1])
        g = gmm_estimate(log_proxy(bars, spec.primary))

        shift = float(np.mean((summ < lo) | (summ > hi)))

        # Parametric bootstrap at the learned estimate.
        boot = [
            _simulate_at(H_net, lam_net, length, prior, spec, rng, net,
                         run_gmm=i < args.gmm_replicates)
            for i in range(args.replicates)
        ]
        bH = np.array([b[0] for b in boot])
        bL = np.array([b[1] for b in boot])

        # Null of exact multifractality, H = 0, at the estimated intermittency.
        null0 = [
            _simulate_at(0.0, lam_net, length, prior, spec, rng, net, run_gmm=False)
            for _ in range(args.replicates)
        ]
        n0 = np.array([b[0] for b in null0])

        # Null of the global rough value reported for indices.
        null1 = [
            _simulate_at(0.10, lam_net, length, prior, spec, rng, net, run_gmm=False)
            for _ in range(args.replicates)
        ]
        n1 = np.array([b[0] for b in null1])

        # Null of monofractality. lambda^2 = 0 is outside the model's support, so
        # the null is simulated at LAMBDA_NULL, small enough that the generalised
        # Hurst function is flat to within Monte Carlo error. Both estimators are
        # scored, because the identification experiment finds lambda^2 recoverable
        # where H is not, which makes this the test the study can actually answer.
        null_lam = [
            _simulate_at(g.H, LAMBDA_NULL, length, prior, spec, rng, net,
                         run_gmm=i < args.gmm_replicates)
            for i in range(args.replicates)
        ]
        nL_net = np.array([b[1] for b in null_lam])
        nL_gmm = np.array([b[3] for b in null_lam])
        nL_gmm = nL_gmm[np.isfinite(nL_gmm)]

        # GMM bootstrap draws, for the interval on the GMM estimate.
        gH = np.array([b[2] for b in boot]); gH = gH[np.isfinite(gH)]
        gL = np.array([b[3] for b in boot]); gL = gL[np.isfinite(gL)]

        entry = {
            "n": length,
            "start": str(frame.index.min().date()),
            "end": str(frame.index.max().date()),
            "lstm": {
                "H": H_net,
                "lambda_sq": lam_net,
                "H_ci95": [float(np.percentile(2 * H_net - bH, 2.5)),
                           float(np.percentile(2 * H_net - bH, 97.5))],
                "lambda_sq_ci95": [float(np.percentile(2 * lam_net - bL, 2.5)),
                                   float(np.percentile(2 * lam_net - bL, 97.5))],
                "bootstrap_sd_H": float(bH.std(ddof=1)),
            },
            "gmm": {
                "H": float(g.H),
                "lambda_sq": float(g.lambda_sq),
                "noise": float(g.noise),
                "H_ci95": [float(np.percentile(2 * g.H - gH, 2.5)),
                           float(np.percentile(2 * g.H - gH, 97.5))] if gH.size else None,
                "lambda_sq_ci95": [
                    float(np.percentile(2 * g.lambda_sq - gL, 2.5)),
                    float(np.percentile(2 * g.lambda_sq - gL, 97.5))] if gL.size else None,
                "bootstrap_sd_lambda_sq": float(gL.std(ddof=1)) if gL.size > 1 else None,
            },
            "test_lambda_eq_0": {
                "null_lambda_sq": LAMBDA_NULL,
                "p_value_gmm": (float(np.mean(nL_gmm >= g.lambda_sq))
                                if nL_gmm.size else None),
                "p_value_lstm": float(np.mean(nL_net >= lam_net)),
                "null_mean_gmm": float(nL_gmm.mean()) if nL_gmm.size else None,
                "null_q95_gmm": (float(np.percentile(nL_gmm, 95))
                                 if nL_gmm.size else None),
                "null_q95_lstm": float(np.percentile(nL_net, 95)),
            },
            "feature_shift_outside_training_band": shift,
            "stale_open_share_before": convention_before,
            "test_H_eq_0": {
                "p_value": float(np.mean(n0 >= H_net)),
                "null_mean": float(n0.mean()),
                "null_q95": float(np.percentile(n0, 95)),
            },
            "test_H_eq_0.10": {
                "p_value": float(2.0 * min(np.mean(n1 >= H_net), np.mean(n1 <= H_net))),
                "null_mean": float(n1.mean()),
            },
            "length_matches_training": bool(abs(length - args.length_label) <= 200),
            "trained_at_length": args.length_label,
        }
        report["indices"][code] = entry
        print(
            f"[{code}] n={length} LSTM H={H_net:.3f} "
            f"[{entry['lstm']['H_ci95'][0]:.3f},{entry['lstm']['H_ci95'][1]:.3f}] "
            f"lam2={lam_net:.3f} | GMM H={g.H:.3f} lam2={g.lambda_sq:.3f} "
            f"| p(H=0)={entry['test_H_eq_0']['p_value']:.3f} "
            f"p(H=0.1)={entry['test_H_eq_0.10']['p_value']:.3f} "
            f"p(lam2=0)={entry['test_lambda_eq_0']['p_value_gmm']} "
            f"| shift={shift:.1%}",
            flush=True,
        )

    report["wall_clock_min"] = (time.time() - t0) / 60.0
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
