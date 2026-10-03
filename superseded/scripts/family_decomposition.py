"""Decompose the estimator margin by training family, on one fixed test set.

The interim report's headline for RQ1 is a pooled margin over a test set that
mixes three families. Two different ways of asking "what survives without the
cascade" can disagree: regenerating a test set from a corpus that excludes the
family changes the paths and the parameter draws as well as the mixture, while
subsetting one test set holds both fixed. Only the second isolates the family.

This scores every model on the SAME held-out paths under the adopted scaling
policy, then bootstraps the margin within each family subset.

Run:
    PYTHONPATH=src python3 scripts/family_decomposition.py \
        --checkpoint baseline=results/lstm_best.pt \
        --checkpoint no_binomial=results_nb3k/lstm_best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rvmf.corpus import CorpusSpec, DEFAULT_Q, fixed_testset
from rvmf.features import FeatureSpec
from rvmf.ghe import anchored_fit
from rvmf.mfdfa import ScalingRange, mfdfa
from rvmf.model import HurstLSTM, ModelConfig, predict_numpy

TEST_SEED = 4837
BOOT_SEED = 4835


def load_model(path: str):
    ckpt = torch.load(path, weights_only=False)
    model = HurstLSTM(ModelConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", action="append", required=True,
                    metavar="NAME=PATH")
    ap.add_argument("--test-size", type=int, default=800)
    ap.add_argument("--length", type=int, default=5000)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--resamples", type=int, default=2000)
    ap.add_argument("--out", default="results/family_decomposition.json")
    args = ap.parse_args()

    torch.set_num_threads(2)
    q = DEFAULT_Q
    fspec = FeatureSpec(stride=args.stride)
    spec = ScalingRange()  # the adopted policy, for both estimators

    tf, ty, tm, tfam, _, tseries = fixed_testset(
        TEST_SEED, CorpusSpec(length=args.length), fspec, args.test_size, q,
        return_series=True,
    )
    fam = np.asarray([str(f) for f in tfam])
    print(f"[setup] test set {args.test_size} paths, families "
          f"{ {k: int((fam == k).sum()) for k in sorted(set(fam))} }", flush=True)

    print(f"[mfdfa] scoring under the '{spec.policy}' policy", flush=True)
    mf = np.empty((args.test_size, q.size))
    for i in range(args.test_size):
        mf[i] = mfdfa(tseries[i], q, 2, spec).h

    preds = {}
    for item in args.checkpoint:
        name, _, path = item.partition("=")
        preds[name] = predict_numpy(load_model(path), tf)
        print(f"[setup] scored {name}", flush=True)

    def rmse(e, idx):
        return float(np.sqrt(((e[idx] - ty[idx]) ** 2 * tm[idx]).sum()
                             / max(tm[idx].sum(), 1)))

    def lam_rmse(e, idx):
        errs = []
        for i in np.flatnonzero(idx):
            if tm[i].sum() < 5:
                continue
            errs.append((anchored_fit(q, e[i]).lam - anchored_fit(q, ty[i]).lam) ** 2)
        return float(np.sqrt(np.mean(errs))) if errs else float("nan")

    subsets = {"all families": np.ones(args.test_size, bool)}
    for k in sorted(set(fam)):
        subsets[f"{k} only"] = fam == k
    subsets["cascade removed"] = fam != "binomial"

    report = {"policy": spec.policy, "test_seed": TEST_SEED,
              "test_size": args.test_size, "subsets": {}}

    for label, idx in subsets.items():
        entry = {"n": int(idx.sum()), "mfdfa": rmse(mf, idx),
                 "mfdfa_lambda": lam_rmse(mf, idx), "models": {}}
        ii = np.flatnonzero(idx)
        for name, pred in preds.items():
            rng = np.random.default_rng(BOOT_SEED)
            margins = []
            for _ in range(args.resamples):
                s = rng.choice(ii, ii.size, replace=True)
                margins.append(rmse(mf, s) - rmse(pred, s))
            lo, hi = np.percentile(margins, [2.5, 97.5])
            entry["models"][name] = {
                "rmse": rmse(pred, idx),
                "lambda_rmse": lam_rmse(pred, idx),
                "margin": float(np.mean(margins)),
                "ci95": [float(lo), float(hi)],
                "straddles_zero": bool(lo < 0 < hi),
            }
        report["subsets"][label] = entry

        line = f"  {label:<18} n={entry['n']:>3}  MF-DFA {entry['mfdfa']:.4f}"
        for name, m in entry["models"].items():
            line += (f" | {name} {m['rmse']:.4f} margin {m['margin']:+.4f} "
                     f"[{m['ci95'][0]:+.4f},{m['ci95'][1]:+.4f}]"
                     f"{'*' if m['straddles_zero'] else ''}")
        print(line, flush=True)

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\n* margin interval includes zero\nwritten to {args.out}")


if __name__ == "__main__":
    main()
