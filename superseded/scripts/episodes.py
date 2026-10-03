"""Regime analysis: does the multifractality strength differ across sub-periods?

RQ4 as originally framed asks whether the spectrum differs between narrow calm
and crisis episodes. That question is not answerable at daily frequency with
this estimator, and the reason is arithmetic rather than empirical: under the
adopted scaling policy a fitted window spanning one decade of scales needs a
ten-segment floor at the top and a smallest scale of ten, so the series must run
to about 1,000 observations. A crisis episode of six to eighteen months supplies
125 to 375. No estimate can be formed there, let alone compared.

The question is therefore asked at the coarsest resolution the estimator
supports: contiguous multi-year sub-periods, each long enough to carry a valid
window, chosen so that each contains a distinct market phase. Every sub-period
is tested against its own null, simulated at that sub-period's length, because
the fabricated width grows as series shorten and a common null would attribute
that growth to the market.

Run:
    PYTHONPATH=src python3 scripts/episodes.py
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

from rvmf.corpus import DEFAULT_Q
from rvmf.data import clean_ohlc, load_ohlc, reconcile
from rvmf.diagnostics import hill_estimator
from rvmf.features import FeatureSpec, extract_features
from rvmf.ghe import anchored_fit, spectrum_width
from rvmf.mfdfa import ScalingRange, mfdfa
from rvmf.model import HurstLSTM, ModelConfig, predict_numpy
from rvmf.nulls import monte_carlo_test
from rvmf.proxies import Estimand, build_estimand

Q = DEFAULT_Q
SEED = 4838

# Contiguous sub-periods, each containing a distinct market phase and each long
# enough to carry a fitted window. Boundaries are set on market history, not on
# the data, so the partition is not chosen to produce a result.
PERIODS = [
    ("2006-01-01", "2009-12-31", "P1 2006-2009, global financial crisis"),
    ("2010-01-01", "2014-12-31", "P2 2010-2014, post-crisis recovery"),
    ("2015-01-01", "2019-12-31", "P3 2015-2019, emerging-market stress"),
    ("2020-01-01", "2026-06-30", "P4 2020-2026, COVID-19 and after"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--checkpoint", default="results/lstm_best.pt")
    ap.add_argument("--out", default="results/episodes.json")
    ap.add_argument("--n-null", type=int, default=200)
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
    fspec = FeatureSpec(stride=10)
    spec = ScalingRange()

    ckpt = torch.load(args.checkpoint, weights_only=False)
    model = HurstLSTM(ModelConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    frame, _ = clean_ohlc(
        load_ohlc(d / f"OHLC_historical_data_{args.primary}.csv")
    )
    cross = d / "Jane_Chipfakacha_20260618235987.xlsx"
    # The secondary vendor file covers J200 only, so it may be applied to J200
    # and to nothing else. Reconciling another index against it would overwrite
    # that index with Top 40 observations.
    reconciled = False
    if cross.exists() and args.primary == "J200":
        sec, _ = clean_ohlc(load_ohlc(cross, sheet="Price Data"))
        frame, _ = reconcile(frame, sec)
        reconciled = True

    est = Estimand(proxy="rv", log=True, convention="series")
    report = {
        "q": Q.tolist(),
        "seed": SEED,
        "index": args.primary,
        "vendor_reconciled": reconciled,
        "n_null": args.n_null,
        "estimand": est.label(),
        "minimum_length_note": (
            "A fitted window spanning one decade of scales requires about 1,000 "
            "observations under the adopted policy, so narrow crisis episodes "
            "cannot be estimated at daily frequency."
        ),
        "periods": [],
    }

    for lo, hi, label in PERIODS:
        sub = frame.loc[lo:hi]
        x = build_estimand(sub, est).to_numpy()
        res = mfdfa(x, Q, 2, spec)
        fit = anchored_fit(Q, res.h)

        feats = extract_features(x, fspec)[None, ...]
        h_ls = predict_numpy(model, feats)[0]
        fit_ls = anchored_fit(Q, h_ls)

        null = monte_carlo_test(
            x, "monofractal", n_replicates=args.n_null, rng=rng, q=Q,
            order=2, scaling_range=spec,
        )

        entry = {
            "label": label,
            "start": lo,
            "end": hi,
            "n": int(x.size),
            "gamma": hill_estimator(np.diff(x)),
            "scaling": res.summary(),
            "mfdfa": {
                "h": res.h.tolist(),
                "H": fit.H,
                "lambda": fit.lam,
                "width": spectrum_width(Q, res.h),
                "anchored_r2": fit.r2,
            },
            "lstm": {"H": fit_ls.H, "lambda": fit_ls.lam,
                     "width": spectrum_width(Q, h_ls)},
            "null": null.summary(),
            "null_widths": [float(w) for w in null.null_widths],
        }
        report["periods"].append(entry)
        print(f"{label}", flush=True)
        print(f"   n={entry['n']:4d}  range [{res.summary()['scale_min']},"
              f"{res.summary()['scale_max']}]  H={fit.H:.3f}  "
              f"lambda={fit.lam:.4f}  width={entry['mfdfa']['width']:.3f}  "
              f"null p={null.p_value:.3f}", flush=True)

    # Is the spread in lambda across periods larger than sampling noise?
    lams = np.array([p["mfdfa"]["lambda"] for p in report["periods"]])
    widths = np.array([p["mfdfa"]["width"] for p in report["periods"]])
    null_sds = np.array([p["null"]["null_sd"] for p in report["periods"]])
    report["dispersion"] = {
        "lambda_min": float(lams.min()),
        "lambda_max": float(lams.max()),
        "lambda_range": float(np.ptp(lams)),
        "lambda_sd": float(lams.std(ddof=1)),
        "width_range": float(np.ptp(widths)),
        "mean_null_sd_of_width": float(null_sds.mean()),
        "width_range_exceeds_null_sd": bool(np.ptp(widths) > null_sds.mean()),
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nlambda across periods: {lams.min():.4f} to {lams.max():.4f} "
          f"(range {np.ptp(lams):.4f}, sd {lams.std(ddof=1):.4f})")
    print(f"width range {np.ptp(widths):.3f} against a mean null sd of "
          f"{null_sds.mean():.3f}")
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
