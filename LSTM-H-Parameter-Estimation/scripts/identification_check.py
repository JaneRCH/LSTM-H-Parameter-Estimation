"""Go/no-go identification check: can daily OHLC place JSE volatility on the
rough-to-multifractal continuum?

Simulates the log S-fBM observation model at known ``(H, lambda^2)``, forms the
daily Garman-Klass proxy exactly as it is formed on the JSE files, and
estimates the parameters back with GMM on the log proxy.  The naive scaling
regression is reported alongside to show the bias it carries.

Decision rule, fixed before the run: if the root mean squared error of ``H``
near ``H = 0.1`` is below 0.05, daily data separates rough from multifractal
and the estimation programme proceeds.  If it is not, the identification
frontier itself becomes the result.

Run:
    PYTHONPATH=src python3 scripts/identification_check.py --reps 60
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from rvmf.diagnostics import pooled_rmse
from rvmf.gmm_lnm import estimate, gjr_regression, default_lags
from rvmf.observation import ObservationParams, simulate_bars
from rvmf.rangeproxy import log_proxy
from rvmf.seeds import generator
from rvmf.sfbm import SfbmParams

H_GRID = (0.0, 0.05, 0.10, 0.20, 0.30)
LAMBDA_GRID = (0.02, 0.05, 0.10)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--length", type=int, default=5100)
    ap.add_argument("--proxy", default="garman_klass")
    ap.add_argument("--steps-per-day", type=int, default=48)
    ap.add_argument("--out", default="results_v2/identification.json")
    args = ap.parse_args()

    rng = generator("identification")
    obs = ObservationParams(steps_per_day=args.steps_per_day)
    rows: list[dict] = []
    t0 = time.time()

    for H in H_GRID:
        for lam in LAMBDA_GRID:
            gmm_H, gmm_lam, gjr_H = [], [], []
            for _ in range(args.reps):
                sf = SfbmParams(
                    H=H,
                    lambda_sq=lam,
                    T=float(rng.uniform(1.0, 10.0)) * args.length,
                    sigma_bar=float(rng.uniform(0.12, 0.35)),
                )
                bars = simulate_bars(args.length, sf, obs, rng)
                y = log_proxy(bars, args.proxy)
                res = estimate(y)
                gmm_H.append(res.H)
                gmm_lam.append(res.lambda_sq)
                gjr_H.append(gjr_regression(y, default_lags(int(0.1 * y.size))))

            gmm_H = np.array(gmm_H)
            gmm_lam = np.array(gmm_lam)
            gjr_H = np.array(gjr_H)
            row = {
                "H_true": H,
                "lambda_sq_true": lam,
                "reps": args.reps,
                "gmm_H_mean": float(gmm_H.mean()),
                "gmm_H_bias": float(gmm_H.mean() - H),
                "gmm_H_sd": float(gmm_H.std(ddof=1)),
                "gmm_H_rmse": float(np.sqrt(np.mean((gmm_H - H) ** 2))),
                "gmm_lambda_mean": float(gmm_lam.mean()),
                "gmm_lambda_rmse": float(np.sqrt(np.mean((gmm_lam - lam) ** 2))),
                "gjr_H_mean": float(gjr_H.mean()),
                "gjr_H_rmse": float(np.sqrt(np.mean((gjr_H - H) ** 2))),
            }
            rows.append(row)
            print(
                f"H={H:.2f} lam2={lam:.2f} | GMM mean {row['gmm_H_mean']:.3f} "
                f"bias {row['gmm_H_bias']:+.3f} rmse {row['gmm_H_rmse']:.3f} "
                f"| GJR mean {row['gjr_H_mean']:.3f} rmse {row['gjr_H_rmse']:.3f} "
                f"| lam2 {row['gmm_lambda_mean']:.3f}",
                flush=True,
            )

    near = [r for r in rows if abs(r["H_true"] - 0.10) < 1e-9]
    verdict = {
        "rmse_H_at_0.10": pooled_rmse(r["gmm_H_rmse"] for r in near),
        "threshold": 0.05,
        "proxy": args.proxy,
        "length": args.length,
        "steps_per_day": args.steps_per_day,
        "wall_clock_min": (time.time() - t0) / 60.0,
    }
    verdict["identified"] = bool(verdict["rmse_H_at_0.10"] < verdict["threshold"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"verdict": verdict, "rows": rows}, indent=2))
    print(f"\nRMSE(H) at H=0.10: {verdict['rmse_H_at_0.10']:.4f} "
          f"(threshold {verdict['threshold']})")
    print(f"IDENTIFIED: {verdict['identified']}")
    print(f"written to {out} in {verdict['wall_clock_min']:.1f} min")


if __name__ == "__main__":
    main()
