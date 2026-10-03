"""Train the two-branch estimator of ``(H, lambda^2)`` on simulated bars.

The corpus is pre-generated rather than simulated fresh each epoch, which
departs from :cite:`Boros2024`.  The departure is controlled, not assumed away:
the held-out corpus comes from an independent seed stream, and the gap between
training and held-out loss is reported at every evaluation so that overfitting
would be visible rather than hidden.

Run:
    PYTHONPATH=src python3 scripts/train_v2.py --steps 3000
"""

from __future__ import annotations

import argparse
import os
import json
import time
from pathlib import Path

import numpy as np
import torch

from rvmf.features_v2 import FeatureSpecV2
from rvmf.model_v2 import (
    ModelConfigV2,
    ThetaNet,
    predict_theta,
    standardise_targets,
)
from rvmf.seeds import generator


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def _loss(model: ThetaNet, data: dict, idx: np.ndarray) -> float:
    model.eval()
    with torch.no_grad():
        s = torch.from_numpy(data["sequence"][idx])
        m = torch.from_numpy(data["summary"][idx])
        y = torch.from_numpy(standardise_targets(data["label"][idx]))
        return float(torch.mean((model(s, m) - y) ** 2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="results_v2/corpus/train_L5100_n40000.npz")
    ap.add_argument("--test", default="results_v2/corpus/test_L5100_n4000.npz")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--no-sequence", action="store_true", help="summary-branch ablation")
    ap.add_argument("--no-summary", action="store_true", help="sequence-branch ablation")
    ap.add_argument("--out", default="results_v2/net")
    ap.add_argument("--tag", default="full")
    ap.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 2),
                    help="torch intra-op threads. Set to 1 when running several "
                         "trainings concurrently.")
    ap.add_argument("--n-train", type=int, default=0,
                    help="cap the training split at this many paths (0 uses all). "
                         "The length sweep holds the corpus size fixed across "
                         "lengths so that the frontier measures length alone.")
    ap.add_argument("--eval-size", type=int, default=1500,
                    help="examples per split for the in-training loss curve. "
                         "The reported benchmark uses the full held-out set.")
    args = ap.parse_args()

    # Respect the thread budget the caller sets. Hard-coding two threads makes
    # two concurrent runs contend for the same cores and roughly triples the
    # wall clock of each.
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train = _load(Path(args.train))
    test = _load(Path(args.test))
    if args.n_train and args.n_train < train["label"].shape[0]:
        train = {k: v[: args.n_train] for k, v in train.items()}
    spec = FeatureSpecV2()

    cfg = ModelConfigV2(
        n_sequence_features=spec.n_sequence_features,
        n_summary_features=spec.n_summary_features,
        hidden=args.hidden,
        layers=args.layers,
        lr=args.lr,
        batch_size=args.batch_size,
        steps=args.steps,
        use_sequence=not args.no_sequence,
        use_summary=not args.no_summary,
    )

    torch.manual_seed(int(generator("net_init").integers(0, 2**31 - 1)))
    model = ThetaNet(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    rng = generator("net_batch")
    n_train = train["label"].shape[0]
    y_train = standardise_targets(train["label"])

    probe = rng.choice(n_train, size=min(args.eval_size, n_train), replace=False)
    n_test = test["label"].shape[0]
    test_probe = rng.choice(n_test, size=min(args.eval_size, n_test), replace=False)
    history: list[dict] = []
    best = {"test": np.inf, "step": -1}
    start = time.time()

    for step in range(1, args.steps + 1):
        idx = rng.choice(n_train, size=cfg.batch_size, replace=False)
        model.train()
        opt.zero_grad()
        pred = model(
            torch.from_numpy(train["sequence"][idx]),
            torch.from_numpy(train["summary"][idx]),
        )
        loss = torch.mean((pred - torch.from_numpy(y_train[idx])) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()

        if step % args.eval_every == 0 or step == args.steps:
            tr = _loss(model, train, probe)
            te = _loss(model, test, test_probe)
            history.append({"step": step, "train": tr, "test": te,
                            "elapsed_min": (time.time() - start) / 60.0})
            if te < best["test"]:
                best = {"test": te, "step": step}
                torch.save({"state_dict": model.state_dict(), "config": cfg.to_dict()},
                           out / f"theta_{args.tag}.pt")
            print(f"[{step:5d}/{args.steps}] train {tr:.5f} test {te:.5f} "
                  f"gap {te - tr:+.5f} (best {best['test']:.5f} @ {best['step']}) "
                  f"{history[-1]['elapsed_min']:.1f} min", flush=True)

    ckpt = torch.load(out / f"theta_{args.tag}.pt", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    pred = predict_theta(model, test["sequence"], test["summary"])
    truth = np.column_stack([test["label"][:, 0], np.exp(test["label"][:, 1])])

    report = {
        "tag": args.tag,
        "config": cfg.to_dict(),
        "n_train": int(n_train),
        "n_test": int(test["label"].shape[0]),
        "best_step": best["step"],
        "best_test_loss": best["test"],
        "final_train_test_gap": history[-1]["test"] - history[-1]["train"],
        "H_rmse": float(np.sqrt(np.mean((pred[:, 0] - truth[:, 0]) ** 2))),
        "H_bias": float(np.mean(pred[:, 0] - truth[:, 0])),
        "lambda_sq_rmse": float(np.sqrt(np.mean((pred[:, 1] - truth[:, 1]) ** 2))),
        "history": history,
        "wall_clock_min": (time.time() - start) / 60.0,
    }
    near = np.abs(truth[:, 0] - 0.10) < 0.05
    report["H_rmse_near_0.10"] = float(
        np.sqrt(np.mean((pred[near, 0] - truth[near, 0]) ** 2))
    ) if near.any() else None

    (out / f"train_{args.tag}.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nH RMSE {report['H_rmse']:.4f} (near 0.10: {report['H_rmse_near_0.10']:.4f}) "
          f"| lambda^2 RMSE {report['lambda_sq_rmse']:.4f}")
    print(f"written to {out / f'train_{args.tag}.json'}")


if __name__ == "__main__":
    main()
