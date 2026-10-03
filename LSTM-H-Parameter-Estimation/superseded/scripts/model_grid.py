"""Estimate h(q) on every index under several trained estimators.

The Monte Carlo nulls and the surrogate decomposition are computed from the
MF-DFA statistic, so they depend on the index and not on the network. Only the
learned profile changes when the training corpus changes. This script therefore
crosses checkpoints with indices cheaply, so the corpus effect and the index
effect can be read apart without re-running the null tests for each model.

Run:
    PYTHONPATH=src python3 scripts/model_grid.py \
        --checkpoint baseline=results/lstm_best.pt \
        --checkpoint no_binomial=results_nb/lstm_best.pt
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
from rvmf.proxies import Estimand, build_estimand

Q = DEFAULT_Q
CODES = ("J200", "J203", "J210", "J213")


def load_model(path: str):
    ckpt = torch.load(path, weights_only=False)
    model = HurstLSTM(ModelConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="a labelled checkpoint; repeat the flag for each model",
    )
    ap.add_argument("--out", default="results/model_grid.json")
    ap.add_argument("--stride", type=int, default=10)
    args = ap.parse_args()

    torch.set_num_threads(2)
    d = Path(args.data_dir)
    fspec = FeatureSpec(stride=args.stride)
    spec = ScalingRange()

    models = {}
    for item in args.checkpoint:
        if "=" not in item:
            raise SystemExit(f"expected NAME=PATH, got {item!r}")
        name, _, path = item.partition("=")
        models[name] = load_model(path)
        print(f"[setup] loaded {name} from {path}", flush=True)

    report = {"q": Q.tolist(), "policy": spec.policy, "indices": {}}
    est = Estimand(proxy="rv", log=True, convention="series")

    for code in CODES:
        frame, _ = clean_ohlc(load_ohlc(d / f"OHLC_historical_data_{code}.csv"))
        # The secondary vendor feed covers J200 only.
        cross = d / "Jane_Chipfakacha_20260618235987.xlsx"
        if code == "J200" and cross.exists():
            sec, _ = clean_ohlc(load_ohlc(cross, sheet="Price Data"))
            frame, _ = reconcile(frame, sec)

        x = build_estimand(frame, est).to_numpy()
        h_mf = mfdfa(x, Q, 2, spec).h
        f_mf = anchored_fit(Q, h_mf)
        entry = {
            "n": int(x.size),
            "start": str(frame.index.min().date()),
            "end": str(frame.index.max().date()),
            "gamma": hill_estimator(np.diff(x)),
            "mfdfa": {
                "H": f_mf.H,
                "lambda": f_mf.lam,
                "width": spectrum_width(Q, h_mf),
                "h": h_mf.tolist(),
            },
            "lstm": {},
        }
        feats = extract_features(x, fspec)[None, ...]
        for name, model in models.items():
            h_ls = predict_numpy(model, feats)[0]
            f_ls = anchored_fit(Q, h_ls)
            entry["lstm"][name] = {
                "H": f_ls.H,
                "lambda": f_ls.lam,
                "width": spectrum_width(Q, h_ls),
                "h": h_ls.tolist(),
            }
        report["indices"][code] = entry

        line = (f"[{code}] n={entry['n']:5d}  MF-DFA H={f_mf.H:.3f} "
                f"lam={f_mf.lam:.4f} w={entry['mfdfa']['width']:.3f}")
        for name in models:
            e = entry["lstm"][name]
            line += (f" | {name} H={e['H']:.3f} lam={e['lambda']:.4f} "
                     f"w={e['width']:.3f}")
        print(line, flush=True)

    Path(args.out).write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
