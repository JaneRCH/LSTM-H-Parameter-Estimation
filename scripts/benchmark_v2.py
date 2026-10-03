"""Score every estimator of ``(H, lambda^2)`` on the same held-out simulated bars.

All estimators meet the identical measurement noise, because they are applied
to the same simulated daily proxies rather than to a latent path.  That is what
makes the comparison fair: the margin of a learned estimator over a classical
one is then a bias correction it has learnt, not an advantage handed to it by a
cleaner input.

Errors are reported as bias, RMSE and the localised relative-error quantiles of
:cite:`Boros2024`, taken in a window around ``H = 0.1`` where the rough
volatility literature places index volatility and where every estimator in that
paper performed worst.

Run:
    PYTHONPATH=src python3 scripts/benchmark_v2.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from rvmf.gmm_lnm import default_lags, estimate, gjr_regression
from rvmf.model_v2 import ModelConfigV2, ThetaNet, predict_theta


def _load_net(path: Path) -> ThetaNet:
    ckpt = torch.load(path, weights_only=False)
    model = ThetaNet(ModelConfigV2(**ckpt["config"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def _metrics(pred: np.ndarray, truth: np.ndarray, window: float = 0.05) -> dict:
    err = pred - truth
    near = np.abs(truth - 0.10) <= window
    out = {
        "bias": float(err.mean()),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "rmse_near_0.10": float(np.sqrt(np.mean(err[near] ** 2))) if near.any() else None,
        "n_near_0.10": int(near.sum()),
    }
    if near.any():
        denom = np.maximum(truth[near], 0.02)
        rel = np.abs(err[near]) / denom
        out["rel_err_q50_near_0.10"] = float(np.quantile(rel, 0.50))
        out["rel_err_q95_near_0.10"] = float(np.quantile(rel, 0.95))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="results_v2/corpus/test_L5100_n4000.npz")
    ap.add_argument("--nets", nargs="*", default=["results_v2/net/theta_full.pt"])
    ap.add_argument("--names", nargs="*", default=["LSTM"])
    ap.add_argument("--classical-subset", type=int, default=1200,
                    help="paths scored by GMM and GJR; both are per-path optimisations")
    ap.add_argument("--out", default="results_v2/benchmark.json")
    args = ap.parse_args()

    torch.set_num_threads(2)
    with np.load(args.test) as z:
        seq, summ, label = z["sequence"], z["summary"], z["label"]
        proxy = z["proxy"] if "proxy" in z.files else None
    H_true = label[:, 0]
    lam_true = np.exp(label[:, 1])

    report: dict = {"test_set": args.test, "n": int(label.shape[0]), "estimators": {}}
    t0 = time.time()

    for name, path in zip(args.names, args.nets):
        p = Path(path)
        if not p.exists():
            print(f"[skip] {name}: {p} not found", flush=True)
            continue
        pred = predict_theta(_load_net(p), seq, summ)
        report["estimators"][name] = {
            "H": _metrics(pred[:, 0], H_true),
            "lambda_sq_rmse": float(np.sqrt(np.mean((pred[:, 1] - lam_true) ** 2))),
        }
        print(f"[{name}] H RMSE {report['estimators'][name]['H']['rmse']:.4f} "
              f"near 0.10 {report['estimators'][name]['H']['rmse_near_0.10']:.4f}", flush=True)

    if proxy is not None:
        k = min(args.classical_subset, proxy.shape[0])
        gmm_H, gmm_lam, gjr_H = [], [], []
        for i in range(k):
            y = proxy[i]
            res = estimate(y)
            gmm_H.append(res.H)
            gmm_lam.append(res.lambda_sq)
            gjr_H.append(gjr_regression(y, default_lags(int(0.1 * y.size))))
            if (i + 1) % 200 == 0:
                print(f"  [classical] {i + 1}/{k}", flush=True)
        report["estimators"]["GMM_lnM"] = {
            "H": _metrics(np.array(gmm_H), H_true[:k]),
            "lambda_sq_rmse": float(np.sqrt(np.mean((np.array(gmm_lam) - lam_true[:k]) ** 2))),
            "n_scored": k,
        }
        report["estimators"]["GJR_regression"] = {
            "H": _metrics(np.array(gjr_H), H_true[:k]),
            "n_scored": k,
        }
    else:
        report["note"] = (
            "The test archive carries no raw proxy series, so the classical "
            "estimators were scored in the identification experiment instead."
        )

    report["wall_clock_min"] = (time.time() - t0) / 60.0
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
