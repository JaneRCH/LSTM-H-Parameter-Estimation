"""Score a trained network against MF-DFA on held-out synthetic paths.

Separated from training so that evaluation settings can be changed without
retraining, and so that the classical benchmark is computed under the policy the
study actually adopts. Scoring against a deliberately weakened classical
baseline would flatter the learned estimator; the default here is the policy
validated in ``scripts/validate.py``.

Run:
    PYTHONPATH=src python3 scripts/benchmark.py --policy adaptive
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rvmf.corpus import CorpusSpec, DEFAULT_Q, fixed_testset, parse_shares
from rvmf.features import FeatureSpec
from rvmf.ghe import anchored_fit
from rvmf.mfdfa import ScalingRange, mfdfa
from rvmf.model import HurstLSTM, ModelConfig, predict_numpy

TEST_SEED = 4837
BOOT_SEED = 4835


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="results/lstm_best.pt")
    ap.add_argument("--out", default="results/benchmark_report.json")
    ap.add_argument("--test-size", type=int, default=800)
    ap.add_argument("--length", type=int, default=5000)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--policy", default="adaptive", choices=["adaptive", "fixed"])
    ap.add_argument("--min-segments", type=int, default=10)
    ap.add_argument("--resamples", type=int, default=2000)
    ap.add_argument(
        "--shares",
        type=str,
        default=None,
        help="corpus mixture for the HELD-OUT TEST SET, same syntax as "
        "train.py. Scoring a model on a test set that drops a family it "
        "never saw is the matched comparison; scoring it on the full "
        "mixture shows what dropping the family cost.",
    )
    args = ap.parse_args()

    torch.set_num_threads(2)
    q = DEFAULT_Q
    cspec = CorpusSpec(length=args.length, shares=parse_shares(args.shares))
    fspec = FeatureSpec(stride=args.stride)
    spec = ScalingRange(policy=args.policy, s_min=10, min_segments=args.min_segments)

    print(f"[setup] building held-out test set (seed {TEST_SEED})", flush=True)
    tf, ty, tm, tfam, _, tseries = fixed_testset(
        TEST_SEED, cspec, fspec, args.test_size, q, return_series=True
    )

    ckpt = torch.load(args.checkpoint, weights_only=False)
    cfg = ModelConfig(**ckpt["config"])
    model = HurstLSTM(cfg)
    model.load_state_dict(ckpt["state_dict"])
    print(f"[setup] loaded {args.checkpoint}", flush=True)

    pred = predict_numpy(model, tf)

    print(f"[mfdfa] scoring under the '{args.policy}' policy", flush=True)
    mf = np.empty_like(pred)
    for i in range(args.test_size):
        mf[i] = mfdfa(tseries[i], q, 2, spec).h

    def rmse_by_q(e):
        num = ((e - ty) ** 2 * tm).sum(axis=0)
        return np.sqrt(num / np.maximum(tm.sum(axis=0), 1))

    def pooled(e, idx=None):
        sel = slice(None) if idx is None else idx
        return float(
            np.sqrt(
                ((e[sel] - ty[sel]) ** 2 * tm[sel]).sum()
                / max(tm[sel].sum(), 1)
            )
        )

    per_family = {}
    for fam in sorted(set(tfam)):
        idx = np.array([f == fam for f in tfam])
        per_family[fam] = {
            "n": int(idx.sum()),
            "lstm": pooled(pred, idx),
            "mfdfa": pooled(mf, idx),
        }

    rng = np.random.default_rng(BOOT_SEED)
    margins = []
    for _ in range(args.resamples):
        s = rng.integers(0, args.test_size, args.test_size)
        margins.append(pooled(mf, s) - pooled(pred, s))
    lo, hi = np.percentile(margins, [2.5, 97.5])

    i2 = int(np.flatnonzero(q == 2.0)[0])
    fits = {"lstm": [], "mfdfa": [], "true": []}
    for i in range(args.test_size):
        if tm[i].sum() < 5 or not tm[i][i2]:
            continue
        fits["lstm"].append(anchored_fit(q, pred[i]))
        fits["mfdfa"].append(anchored_fit(q, mf[i]))
        fits["true"].append(anchored_fit(q, ty[i]))

    err = lambda k, attr: float(
        np.sqrt(
            np.mean(
                [
                    (getattr(a, attr) - getattr(b, attr)) ** 2
                    for a, b in zip(fits[k], fits["true"])
                ]
            )
        )
    )

    report = {
        "policy": args.policy,
        "min_segments": args.min_segments,
        "test_size": args.test_size,
        "test_seed": TEST_SEED,
        "config": ckpt["config"],
        "feature_spec": {
            "windows": list(fspec.windows),
            "orders": list(fspec.orders),
            "stride": fspec.stride,
            "standardise": fspec.standardise,
            "n_features": int(tf.shape[2]),
            "sequence_length": int(tf.shape[1]),
        },
        "corpus": {
            "shares": cspec.shares,
            "length": cspec.length,
            "band": list(cspec.band),
        },
        "q": q.tolist(),
        "rmse_by_q": {
            "lstm": rmse_by_q(pred).tolist(),
            "mfdfa": rmse_by_q(mf).tolist(),
        },
        "rmse_pooled": {"lstm": pooled(pred), "mfdfa": pooled(mf)},
        "per_family": per_family,
        "paired_bootstrap_margin": {
            "mean": float(np.mean(margins)),
            "ci95": [float(lo), float(hi)],
            "resamples": args.resamples,
        },
        "anchored_rmse": {
            "H": {"lstm": err("lstm", "H"), "mfdfa": err("mfdfa", "H")},
            "lambda": {"lstm": err("lstm", "lam"), "mfdfa": err("mfdfa", "lam")},
            "n": len(fits["true"]),
        },
    }

    # Carry across the training provenance so the report cites one file.
    train_path = Path("results/train_report.json")
    if train_path.exists():
        t = json.loads(train_path.read_text())
        report["best_step"] = t.get("best_step")
        report["best_val_loss"] = t.get("best_val_loss")
        report["history"] = t.get("history")
        report["corpus"]["paths_seen"] = t.get("corpus", {}).get("paths_seen")
        report["seeds"] = t.get("seeds")

    Path(args.out).write_text(json.dumps(report, indent=2))

    print(f"\n=== held-out benchmark ('{args.policy}' policy) ===")
    print(f"pooled RMSE   LSTM {report['rmse_pooled']['lstm']:.4f}   "
          f"MF-DFA {report['rmse_pooled']['mfdfa']:.4f}")
    print(f"margin {np.mean(margins):.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
    for fam, r in per_family.items():
        print(f"  {fam:18s} n={r['n']:4d}  LSTM {r['lstm']:.4f}  "
              f"MF-DFA {r['mfdfa']:.4f}")
    a = report["anchored_rmse"]
    print(f"anchored H      RMSE  LSTM {a['H']['lstm']:.4f}  "
          f"MF-DFA {a['H']['mfdfa']:.4f}")
    print(f"anchored lambda RMSE  LSTM {a['lambda']['lstm']:.4f}  "
          f"MF-DFA {a['lambda']['mfdfa']:.4f}")
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
