#!/usr/bin/env python3
"""Power of the test of monofractality as a function of window length.

The companion to ``scripts/power_study.py``.  That script asks how precisely
each parameter is recovered.  This one asks a different question about the same
windows, and the two can disagree.

The report rejects monofractality on every index by comparing the estimated
intermittency against the distribution the estimator produces when the process
is not multifractal, simulated at the same length.  Both sides of that
comparison move as the window shortens: the estimate becomes noisier, and the
null distribution widens, because a shorter series fabricates more apparent
intermittency.  Whether the test still separates is therefore not implied by the
estimator's error and has to be measured.

For each length the null is simulated at ``LAMBDA_NULL``, the same value the
empirical tests use, and its ninety-fifth percentile is taken as the critical
value.  The alternative is simulated at the intermittencies the four indices
carry.  Power is the share of replicates exceeding the critical value, which is
the probability of detecting multifractality of that strength in a window of
that length.

The estimator at each length is the one the length sweep trained, so the two
studies describe the same estimator.

Run:
    PYTHONPATH=src python3 scripts/power_lambda.py
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
from rvmf.features_v2 import FeatureSpecV2, extract
from rvmf.model_v2 import ModelConfigV2, ThetaNet, predict_theta
from rvmf.observation import simulate_bars
from rvmf.seeds import generator
from rvmf.sfbm import SfbmParams

#: Identical to scripts/empirical_v2.py. Zero is outside the support of the
#: model, so the monofractal null is simulated just inside it.
LAMBDA_NULL: float = 1e-4

#: The intermittencies the four indices carry, lowest, middle and highest.
LAMBDA_ALT: tuple[float, ...] = (0.017, 0.030, 0.047)

#: H is held at the centre of the empirical range. Power depends on it, so the
#: dependence is reported rather than averaged away.
H_FIXED: float = 0.10

LEVEL: float = 0.05


def load_net(path: Path) -> ThetaNet:
    ckpt = torch.load(path, weights_only=False)
    net = ThetaNet(ModelConfigV2(**ckpt["config"]))
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net


def draw(H: float, lam: float, length: int, prior: PriorSpec,
         spec: FeatureSpecV2, rng: np.random.Generator, net: ThetaNet) -> float:
    """One replicate: simulate at fixed targets, return the estimated lambda^2."""
    _, obs = draw_parameters(length, prior, rng)
    sf = SfbmParams(
        H=H, lambda_sq=lam,
        T=float(np.exp(rng.uniform(*np.log(prior.T_over_L)))) * length,
        sigma_bar=float(rng.uniform(*prior.sigma_bar)),
    )
    bars = simulate_bars(length, sf, obs, rng)
    seq, summ = extract(bars, spec)
    return float(predict_theta(net, seq[None, ...], summ[None, ...])[0][1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lengths", type=int, nargs="*",
                    default=[250, 500, 1000, 2000, 3500, 5100])
    ap.add_argument("--null-reps", type=int, default=400)
    ap.add_argument("--alt-reps", type=int, default=300)
    ap.add_argument("--net-dir", default="results_v2/net")
    ap.add_argument("--out", default="results_v2/power_study/power_lambda.json")
    args = ap.parse_args()

    prior, spec = PriorSpec(), FeatureSpecV2()
    rows: list[dict] = []
    t0 = time.time()

    for length in args.lengths:
        net_path = Path(args.net_dir) / f"theta_pw{length}.pt"
        if not net_path.exists():
            print(f"L={length}: no estimator at {net_path}, skipped", flush=True)
            continue
        net = load_net(net_path)
        # One registered stream, an independent replicate per length, so a
        # single length can be re-run without changing the others' draws.
        rng = generator("power_lambda", replicate=length)

        null = np.array([draw(H_FIXED, LAMBDA_NULL, length, prior, spec, rng, net)
                         for _ in range(args.null_reps)])
        critical = float(np.quantile(null, 1.0 - LEVEL))

        row = {"length": length, "null_median": float(np.median(null)),
               "critical_value": critical, "null_reps": args.null_reps,
               "alt_reps": args.alt_reps, "power": {}}
        for lam in LAMBDA_ALT:
            alt = np.array([draw(H_FIXED, lam, length, prior, spec, rng, net)
                            for _ in range(args.alt_reps)])
            row["power"][f"{lam:.3f}"] = {
                "power": float(np.mean(alt > critical)),
                "mean_estimate": float(alt.mean()),
            }
        rows.append(row)
        print(f"L={length:5d}  critical={critical:.4f}  "
              + "  ".join(f"power({k})={v['power']:.2f}"
                          for k, v in row["power"].items()), flush=True)

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            {"H_fixed": H_FIXED, "lambda_null": LAMBDA_NULL, "level": LEVEL,
             "wall_clock_min": (time.time() - t0) / 60.0, "rows": rows},
            indent=2) + "\n")


if __name__ == "__main__":
    main()
