#!/usr/bin/env python3
"""Map the latent parameters to the generalised Hurst function a scaling
estimator would report.

The supervisor's question, and the title of this study, are posed in terms of
the generalised Hurst function ``H(q)``: multifractality is the statement that
``H(q)`` varies with ``q``.  The model estimated here is parameterised by
``(H, lambda^2)`` instead.  This script establishes the correspondence
empirically rather than by quoting an asymptotic formula, because over the
finite range of scales a daily series supports the asymptotic result is not
what an estimator returns.

Which object the exponent is measured on decides the answer, and that is the
first result this script produces.  The latent log volatility ``omega`` is a
Gaussian process.  Every Gaussian process is monofractal: all of its structure
functions share one scaling exponent, so ``H(q)`` is flat in ``q`` whatever
``lambda^2`` is.  The multifractality of the log S-fBM lives in the measure
``M_tau = int exp(omega)``, and taking a logarithm of that measure linearises it
away.  A scaling estimator applied to a *log* volatility series is therefore
measuring a monofractal object by construction, and any curvature it reports is
a finite-sample artefact rather than a property of the process.

Three objects are measured for each ``(H, lambda^2)`` on a grid:

1. ``omega`` itself, the latent log volatility, as the monofractal control;
2. the integrated variance measure aggregated over ``tau`` days, which is where
   the multifractality of the model resides;
3. the daily returns the model produces, which is what an analyst observes.

For each, the structure functions ``S_q(tau) = E[|increment|^q]`` are formed over
the scale range a daily series supports and ``H(q)`` is the slope of
``log S_q`` against ``log tau``, divided by ``q``.  The reported quantity is the
slope of ``H(q)`` in ``q``: zero for a monofractal series, negative for a
multifractal one.

Run:
    PYTHONPATH=src python3 scripts/apparent_hq.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rvmf.observation import ObservationParams, simulate_bars
from rvmf.seeds import generator
from rvmf.sfbm import SfbmParams, simulate_omega

Q_GRID = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
H_GRID = (0.00, 0.05, 0.08, 0.10, 0.20, 0.30)
LAMBDA_GRID = (0.0001, 0.008, 0.020, 0.035, 0.050, 0.100)
# Scales a daily series of ~5,100 observations actually supports: below about a
# week the proxy is noise dominated, above about a year there are too few
# independent increments for the moment to be estimated.
SCALES = np.unique(np.round(np.logspace(np.log10(5), np.log10(250), 14)).astype(int))


def hq_from_series(series: np.ndarray, q_grid: np.ndarray,
                   scales: np.ndarray) -> np.ndarray:
    """H(q) of a stack of series, from increments at each scale."""
    log_s = np.log(scales.astype(float))
    out = np.empty(q_grid.size)
    for i, q in enumerate(q_grid):
        sq = np.array([
            np.mean(np.abs(series[:, s:] - series[:, :-s]) ** q) for s in scales
        ])
        out[i] = np.polyfit(log_s, np.log(sq), 1)[0] / q
    return out


def aggregate_measure(iv: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Cumulative integrated variance, so that differencing at lag tau gives M_tau."""
    return np.cumsum(iv, axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=24)
    ap.add_argument("--length", type=int, default=5100)
    ap.add_argument("--out", default="results_v2/apparent_hq.json")
    args = ap.parse_args()

    rng = generator("identification")
    obs = ObservationParams()
    rows = []
    print(f"{'H':>5}{'lam2':>8} | {'omega':>18} | {'measure':>18} | {'returns':>18}")
    print(f"{'':>5}{'':>8} | {'H(2)':>8}{'dH/dq':>10} | {'H(2)':>8}{'dH/dq':>10} "
          f"| {'H(2)':>8}{'dH/dq':>10}")
    for H in H_GRID:
        for lam in LAMBDA_GRID:
            p = SfbmParams(H=H, lambda_sq=lam, T=4.0 * args.length)
            om, meas, ret = [], [], []
            for _ in range(args.reps):
                bars = simulate_bars(args.length, p, obs, rng)
                om.append(simulate_omega(args.length, 1.0, p, rng))
                meas.append(np.cumsum(bars["integrated_variance"]))
                ret.append(np.cumsum(bars["ret"]))
            hq = {
                "omega": hq_from_series(np.stack(om), Q_GRID, SCALES),
                "measure": hq_from_series(np.stack(meas), Q_GRID, SCALES),
                "returns": hq_from_series(np.stack(ret), Q_GRID, SCALES),
            }
            row = dict(H=H, lambda_sq=lam)
            for name, v in hq.items():
                row[f"hq_{name}"] = v.tolist()
                row[f"slope_{name}"] = float(np.polyfit(Q_GRID, v, 1)[0])
            rows.append(row)
            print(f"{H:5.2f}{lam:8.4f} | "
                  + " | ".join(f"{hq[k][3]:8.3f}{row['slope_' + k]:+10.4f}"
                               for k in ("omega", "measure", "returns")))

    out = dict(q_grid=Q_GRID.tolist(), scales=SCALES.tolist(),
               reps=args.reps, length=args.length, rows=rows)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
