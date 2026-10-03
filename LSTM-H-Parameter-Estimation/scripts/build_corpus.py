"""Pre-generate a labelled corpus of simulated daily bars.

Writes a compressed archive holding both input branches and the latent
``(H, log lambda^2)`` label for every path, together with the prior and the
feature specification used to build it.

Run:
    PYTHONPATH=src python3 scripts/build_corpus.py --n 40000 --split train
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from rvmf.corpus_v2 import PriorSpec, build_corpus
from rvmf.features_v2 import FeatureSpecV2
from rvmf.seeds import generator


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--length", type=int, default=5100)
    ap.add_argument("--split", default="train", choices=("train", "test", "val"))
    ap.add_argument("--out-dir", default="results_v2/corpus")
    ap.add_argument(
        "--with-proxy",
        action="store_true",
        help="also store the daily log proxy series for each path. Needed by the "
             "held-out split, because the moment and regression benchmarks are "
             "applied to the series itself rather than to the extracted features.",
    )
    args = ap.parse_args()

    spec = FeatureSpecV2()
    prior = PriorSpec()
    rng = generator(f"corpus_{args.split}")

    t0 = time.time()
    data = build_corpus(args.n, args.length, prior, spec, rng,
                        progress=max(1, args.n // 20), with_proxy=args.with_proxy)
    elapsed = (time.time() - t0) / 60.0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.split}_L{args.length}_n{args.n}"
    np.savez_compressed(out_dir / f"{stem}.npz", **data)

    meta = {
        "split": args.split,
        "n": args.n,
        "length": args.length,
        "seed_stream": f"corpus_{args.split}",
        "with_proxy": args.with_proxy,
        "prior": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(prior).items()},
        "feature_spec": {
            "block": spec.block,
            "channels": list(spec.channels),
            "primary": spec.primary,
            "orders": list(spec.orders),
            "deltas": spec.deltas.tolist(),
            "lags": spec.lags.tolist(),
            "n_sequence_features": spec.n_sequence_features,
            "n_summary_features": spec.n_summary_features,
        },
        "shapes": {k: list(v.shape) for k, v in data.items()},
        "wall_clock_min": elapsed,
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwritten {out_dir / f'{stem}.npz'} in {elapsed:.1f} min")


if __name__ == "__main__":
    main()
