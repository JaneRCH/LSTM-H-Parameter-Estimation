"""Train the vector-output LSTM and benchmark it against MF-DFA.

Batches are simulated fresh at every step, so the effective corpus is
``steps * batch_size`` distinct paths and there is no fixed training set to
overfit.  Evaluation and test sets are fixed by seed.

Run:
    PYTHONPATH=src python3 scripts/train.py --steps 2500
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from rvmf.corpus import CorpusSpec, DEFAULT_Q, fixed_testset, make_batch, parse_shares
from rvmf.features import FeatureSpec
from rvmf.ghe import anchored_fit, spectrum_width
from rvmf.mfdfa import ScalingRange, mfdfa
from rvmf.model import HurstLSTM, ModelConfig, masked_mse, predict_numpy

SEEDS = {"corpus": 4832, "val": 4836, "test": 4837, "init": 4833}


def evaluate(model, feats, labels, mask):
    pred = predict_numpy(model, feats)
    err = (pred - labels) ** 2
    denom = max(mask.sum(), 1)
    return float((err * mask).sum() / denom), pred


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--length", type=int, default=5000)
    ap.add_argument("--val-size", type=int, default=256)
    ap.add_argument("--test-size", type=int, default=800)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--pooling", type=str, default="mean")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--standardise", type=str, default="per_order")
    ap.add_argument("--out", type=str, default="results")
    ap.add_argument(
        "--shares",
        type=str,
        default=None,
        help="override the corpus family mixture, e.g. "
        "'fbm=0.375,lognormal_mmar=0.625'. Shares are renormalised; families "
        "not named are set to zero. Omit for the default mixture.",
    )
    args = ap.parse_args()

    torch.set_num_threads(2)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    q = DEFAULT_Q
    cspec = CorpusSpec(length=args.length, shares=parse_shares(args.shares))
    fspec = FeatureSpec(stride=args.stride, standardise=args.standardise)

    print(f"[setup] building fixed evaluation sets", flush=True)
    t0 = time.time()
    vf, vy, vm, vfam, _ = fixed_testset(SEEDS["val"], cspec, fspec, args.val_size, q)
    tf, ty, tm, tfam, tpar, tseries = fixed_testset(
        SEEDS["test"], cspec, fspec, args.test_size, q, return_series=True
    )
    print(f"[setup] done in {time.time()-t0:.1f}s; seq={vf.shape[1]} feat={vf.shape[2]}", flush=True)

    cfg = ModelConfig(
        n_features=vf.shape[2],
        n_outputs=q.size,
        hidden=args.hidden,
        layers=args.layers,
        pooling=args.pooling,
        batch_size=args.batch_size,
        steps=args.steps,
        lr=args.lr,
    )
    torch.manual_seed(SEEDS["init"])
    model = HurstLSTM(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    rng = np.random.default_rng(SEEDS["corpus"])
    history: list[dict] = []
    best = {"val": np.inf, "step": -1}
    start = time.time()

    for step in range(1, args.steps + 1):
        f, y, m, _, _ = make_batch(rng, cspec, fspec, args.batch_size, q)
        model.train()
        opt.zero_grad()
        loss = masked_mse(
            model(torch.from_numpy(f)),
            torch.from_numpy(y),
            torch.from_numpy(m),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()

        if step % args.eval_every == 0 or step == args.steps:
            vloss, _ = evaluate(model, vf, vy, vm)
            rec = {
                "step": step,
                "train_loss": float(loss.item()),
                "val_loss": vloss,
                "elapsed_s": time.time() - start,
            }
            history.append(rec)
            if vloss < best["val"]:
                best = {"val": vloss, "step": step}
                torch.save(
                    {"state_dict": model.state_dict(), "config": cfg.to_dict()},
                    out / "lstm_best.pt",
                )
            print(
                f"[{step:5d}/{args.steps}] train {loss.item():.5f} "
                f"val {vloss:.5f} (best {best['val']:.5f} @ {best['step']}) "
                f"{rec['elapsed_s']/60:.1f} min",
                flush=True,
            )

    # ---- Final benchmark on the held-out test set ----------------------------
    print("[test] loading best checkpoint and scoring", flush=True)
    ckpt = torch.load(out / "lstm_best.pt", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    _, pred = evaluate(model, tf, ty, tm)

    print("[test] running MF-DFA benchmark on the same paths", flush=True)
    sr = ScalingRange(policy="fixed", s_min=10, min_segments=10)
    mf = np.empty_like(pred)
    for i in range(args.test_size):
        mf[i] = mfdfa(tseries[i], q, 2, sr).h

    def rmse_by_q(estimate):
        num = ((estimate - ty) ** 2 * tm).sum(axis=0)
        den = np.maximum(tm.sum(axis=0), 1)
        return np.sqrt(num / den)

    lstm_q, mfdfa_q = rmse_by_q(pred), rmse_by_q(mf)
    pooled = lambda e: float(
        np.sqrt(((e - ty) ** 2 * tm).sum() / max(tm.sum(), 1))
    )

    per_family = {}
    for fam in sorted(set(tfam)):
        idx = np.array([f == fam for f in tfam])
        sub = lambda e: float(
            np.sqrt(
                ((e[idx] - ty[idx]) ** 2 * tm[idx]).sum()
                / max(tm[idx].sum(), 1)
            )
        )
        per_family[fam] = {
            "n": int(idx.sum()),
            "lstm": sub(pred),
            "mfdfa": sub(mf),
        }

    # Paired bootstrap on the pooled margin.
    boot_rng = np.random.default_rng(4835)
    margins = []
    for _ in range(2000):
        s = boot_rng.integers(0, args.test_size, args.test_size)
        e_l = np.sqrt(
            ((pred[s] - ty[s]) ** 2 * tm[s]).sum() / max(tm[s].sum(), 1)
        )
        e_m = np.sqrt(((mf[s] - ty[s]) ** 2 * tm[s]).sum() / max(tm[s].sum(), 1))
        margins.append(e_m - e_l)
    lo, hi = np.percentile(margins, [2.5, 97.5])

    # Anchored (H, lambda) recovery, derived from the same predicted vectors.
    anch = {"lstm": [], "mfdfa": [], "true": []}
    for i in range(args.test_size):
        valid = tm[i]
        if valid.sum() < 5 or not valid[list(q).index(2.0)]:
            continue
        anch["lstm"].append(anchored_fit(q, pred[i]))
        anch["mfdfa"].append(anchored_fit(q, mf[i]))
        anch["true"].append(anchored_fit(q, ty[i]))
    lam_err = lambda k: float(
        np.sqrt(
            np.mean(
                [(a.lam - b.lam) ** 2 for a, b in zip(anch[k], anch["true"])]
            )
        )
    )
    big_h_err = lambda k: float(
        np.sqrt(
            np.mean([(a.H - b.H) ** 2 for a, b in zip(anch[k], anch["true"])])
        )
    )

    report = {
        "config": cfg.to_dict(),
        "feature_spec": {
            "windows": list(fspec.windows),
            "orders": list(fspec.orders),
            "stride": fspec.stride,
            "standardise": fspec.standardise,
            "n_features": vf.shape[2],
            "sequence_length": int(vf.shape[1]),
        },
        "corpus": {
            "shares": cspec.shares,
            "length": cspec.length,
            "paths_seen": args.steps * args.batch_size,
            "band": list(cspec.band),
        },
        "seeds": SEEDS,
        "best_step": best["step"],
        "best_val_loss": best["val"],
        "q": q.tolist(),
        "rmse_by_q": {"lstm": lstm_q.tolist(), "mfdfa": mfdfa_q.tolist()},
        "rmse_pooled": {"lstm": pooled(pred), "mfdfa": pooled(mf)},
        "per_family": per_family,
        "paired_bootstrap_margin": {
            "mean": float(np.mean(margins)),
            "ci95": [float(lo), float(hi)],
            "resamples": 2000,
        },
        "anchored_rmse": {
            "H": {"lstm": big_h_err("lstm"), "mfdfa": big_h_err("mfdfa")},
            "lambda": {"lstm": lam_err("lstm"), "mfdfa": lam_err("mfdfa")},
            "n": len(anch["true"]),
        },
        "history": history,
        "wall_clock_min": (time.time() - start) / 60.0,
    }
    (out / "train_report.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(
        out / "test_predictions.npz",
        pred=pred,
        mfdfa=mf,
        truth=ty,
        mask=tm,
        families=np.array(tfam),
        q=q,
    )

    print("\n=== held-out benchmark ===", flush=True)
    print(f"pooled RMSE   LSTM {report['rmse_pooled']['lstm']:.4f}   "
          f"MF-DFA {report['rmse_pooled']['mfdfa']:.4f}", flush=True)
    print(f"margin {np.mean(margins):.4f}  95% CI [{lo:.4f}, {hi:.4f}]", flush=True)
    for fam, r in per_family.items():
        print(f"  {fam:20s} n={r['n']:4d}  LSTM {r['lstm']:.4f}  MF-DFA {r['mfdfa']:.4f}",
              flush=True)
    print(f"anchored lambda RMSE  LSTM {lam_err('lstm'):.4f}  "
          f"MF-DFA {lam_err('mfdfa'):.4f}", flush=True)
    print(f"anchored H      RMSE  LSTM {big_h_err('lstm'):.4f}  "
          f"MF-DFA {big_h_err('mfdfa'):.4f}", flush=True)


if __name__ == "__main__":
    main()
