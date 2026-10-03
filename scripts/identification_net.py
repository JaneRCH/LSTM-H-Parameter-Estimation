#!/usr/bin/env python3
"""Evaluate a trained estimator on the fixed-parameter grid of the
identification experiment, so it is comparable with the moment estimator.

Why this is not the same as the held-out test error
---------------------------------------------------
The held-out test set is drawn from the prior the network was trained on, so a
test error measures Bayes risk under that prior. An estimator can have low Bayes
risk while being unable to separate two nearby parameter values, because the
prior carries much of the information. The pre-registered rule concerns risk at
a point: how precisely H is recovered when H is held fixed and only the nuisance
parameters vary. That is what this script measures, on the same grid, the same
replicate count and the same observation model as
``scripts/identification_check.py``, so the two are directly comparable.

Run:
    PYTHONPATH=src python3 scripts/identification_net.py --net results_v2/net/theta_full.pt
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
from rvmf.diagnostics import pooled_rmse
from rvmf.features_v2 import FeatureSpecV2, extract
from rvmf.model_v2 import ModelConfigV2, ThetaNet, predict_theta
from rvmf.observation import simulate_bars
from rvmf.seeds import generator
from rvmf.sfbm import SfbmParams

# Identical to scripts/identification_check.py.
H_GRID = (0.0, 0.05, 0.10, 0.20, 0.30)
LAMBDA_GRID = (0.02, 0.05, 0.10)


def load_net(path: Path) -> ThetaNet:
    ckpt = torch.load(path, weights_only=False)
    net = ThetaNet(ModelConfigV2(**ckpt["config"]))
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="results_v2/net/theta_full.pt")
    ap.add_argument("--name", default="LSTM")
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--length", type=int, default=5100)
    ap.add_argument("--out", default="results_v2/identification_net.json")
    args = ap.parse_args()

    net = load_net(Path(args.net))
    spec = FeatureSpecV2()
    prior = PriorSpec()
    # The same stream as the moment-estimator run, so the two see the same
    # nuisance draws and the comparison carries no sampling difference.
    rng = generator("identification")
    rows: list[dict] = []
    t0 = time.time()

    for H in H_GRID:
        for lam in LAMBDA_GRID:
            est_H, est_lam = [], []
            for _ in range(args.reps):
                _, obs = draw_parameters(args.length, prior, rng)
                sf = SfbmParams(
                    H=H, lambda_sq=lam,
                    T=float(rng.uniform(1.0, 10.0)) * args.length,
                    sigma_bar=float(rng.uniform(0.12, 0.35)),
                )
                bars = simulate_bars(args.length, sf, obs, rng)
                seq, summ = extract(bars, spec)
                theta = predict_theta(net, seq[None, ...], summ[None, ...])[0]
                est_H.append(float(theta[0]))
                est_lam.append(float(theta[1]))

            eH, eL = np.array(est_H), np.array(est_lam)
            rows.append(dict(
                H_true=H, lambda_sq_true=lam, reps=args.reps,
                H_mean=float(eH.mean()), H_bias=float(eH.mean() - H),
                H_sd=float(eH.std(ddof=1)),
                H_rmse=float(np.sqrt(np.mean((eH - H) ** 2))),
                lambda_mean=float(eL.mean()),
                lambda_rmse=float(np.sqrt(np.mean((eL - lam) ** 2))),
            ))
            r = rows[-1]
            print(f"H={H:.2f} lam2={lam:.2f} | {args.name} mean {r['H_mean']:.3f} "
                  f"bias {r['H_bias']:+.3f} rmse {r['H_rmse']:.3f} sd {r['H_sd']:.3f} "
                  f"| lam2 {r['lambda_mean']:.3f} rmse {r['lambda_rmse']:.3f}", flush=True)

    at_target = next(r for r in rows if r["H_true"] == 0.10 and r["lambda_sq_true"] == 0.05)
    pooled = pooled_rmse(r["H_rmse"] for r in rows if r["H_true"] == 0.10)
    out = dict(
        estimator=args.name, net=args.net,
        verdict=dict(rmse_H_at_0_10=pooled, threshold=0.05, identified=pooled < 0.05,
                     length=args.length, wall_clock_min=(time.time() - t0) / 60.0),
        rows=rows,
    )
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nRMSE(H) at H=0.10, pooled over lambda^2: {pooled:.4f} (threshold 0.05)")
    print(f"IDENTIFIED: {pooled < 0.05}")
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
