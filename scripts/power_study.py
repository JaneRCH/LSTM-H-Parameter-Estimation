#!/usr/bin/env python3
"""The identification frontier as a function of window length.

Why this exists
---------------
The identification experiment of ``scripts/identification_check.py`` was run at
one length, 5,100 days, the length of the longest index.  Every statement the
report makes about what can and cannot be estimated is therefore a statement
about that length.  A sub-period analysis asks the same questions of windows of
250 to 500 days, where nothing has been measured.  Reporting two sub-period
estimates and calling their difference a regime change, without knowing the
estimator's error at that length, is the failure the rest of this study exists
to expose in scaling estimators.

So the frontier is measured first.  For each length the whole chain is rebuilt:
a training corpus at that length, an estimator trained on it, and the same
fixed-parameter grid used at 5,100, with the moment estimator alongside.  The
output answers one question per parameter.  For H: the shortest window at which
the root mean squared error near H = 0.1 falls below the threshold of 0.05 that
was fixed before any of this was run.  For lambda squared: the shortest window
at which the error is a usable fraction of the quantity being estimated.

Three design points
-------------------
The corpus size is held fixed across lengths with ``--n-train``, so a change in
error is a change in what the window carries and not in how much training data
was available.

The learned estimator here is the summary-branch variant.  It trains in about
eight seconds against roughly fifty minutes for the full network, which is what
makes a six-point sweep affordable, and its error at 5,100 is 0.034 against
0.032 for the full network.  The frontier it produces is therefore slightly
conservative: the full network would reach any given error at a shorter window.

Every length draws from the same ``identification`` seed stream, so the lengths
are paired rather than independent, and a difference between two lengths is not
a difference in the draw.

Run:
    PYTHONPATH=src nohup python3 scripts/power_study.py > results_v2/power_study.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from rvmf.diagnostics import pooled_rmse

#: Window lengths in trading days.  250 is one year, the scale of a crisis
#: episode; 5,100 is the longest index and reproduces the published result.
LENGTHS: tuple[int, ...] = (250, 500, 1000, 2000, 3500, 5100)

#: Held fixed across lengths so the sweep isolates the window.
N_TRAIN: int = 20000
N_TEST: int = 2000

CORPUS = Path("results_v2/corpus")
NET = Path("results_v2/net")
OUT = Path("results_v2/power_study")


def run(cmd: list[str], label: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {label}", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:], flush=True)
        print(r.stderr[-3000:], file=sys.stderr, flush=True)
        raise SystemExit(f"failed: {label}")
    print(f"           done in {(time.time()-t0)/60:.1f} min", flush=True)


def corpus(split: str, length: int, n: int, with_proxy: bool) -> Path:
    """Build the corpus for this split and length if it is not already there.

    An existing corpus of at least ``n`` paths is reused and capped at training
    time, so the 5,100 and 3,500 corpora built for the main results are not
    rebuilt.
    """
    exact = CORPUS / f"{split}_L{length}_n{n}.npz"
    if exact.exists():
        return exact
    for candidate in sorted(CORPUS.glob(f"{split}_L{length}_n*.npz")):
        size = int(candidate.stem.split("_n")[-1])
        if size >= n:
            return candidate
    cmd = ["python3", "scripts/build_corpus.py", "--n", str(n),
           "--length", str(length), "--split", split]
    if with_proxy:
        cmd.append("--with-proxy")
    run(cmd, f"corpus {split} L={length} n={n}")
    return exact


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lengths", type=int, nargs="*", default=list(LENGTHS))
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--steps", type=int, default=3000)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    summary: list[dict] = []

    for length in args.lengths:
        tag = f"pw{length}"
        train_path = corpus("train", length, N_TRAIN, with_proxy=False)
        test_path = corpus("test", length, N_TEST, with_proxy=True)

        run(["python3", "scripts/train_v2.py",
             "--train", str(train_path), "--test", str(test_path),
             "--n-train", str(N_TRAIN), "--steps", str(args.steps),
             "--no-sequence", "--tag", tag, "--threads", "2"],
            f"train summaries-only L={length}")

        run(["python3", "scripts/identification_net.py",
             "--net", str(NET / f"theta_{tag}.pt"), "--name", "Learned",
             "--reps", str(args.reps), "--length", str(length),
             "--out", str(OUT / f"net_L{length}.json")],
            f"identification, learned, L={length}")

        run(["python3", "scripts/identification_check.py",
             "--reps", str(args.reps), "--length", str(length),
             "--out", str(OUT / f"gmm_L{length}.json")],
            f"identification, moments, L={length}")

        summary = write_frontier(args.reps)
        print(f"           L={length}: {summary[-1]}", flush=True)

    print("sweep complete", flush=True)


def _pool(rows: list[dict], key: str, where) -> float | None:
    """Pool the per-cell errors over the matching cells.

    Delegates to :func:`rvmf.diagnostics.pooled_rmse`, the one rule the whole
    study uses, so a figure here is read against the same threshold as the
    figures in Section 4.
    """
    vals = [r[key] for r in rows if where(r) and r.get(key) is not None]
    return pooled_rmse(vals) if vals else None


def write_frontier(reps: int) -> list[dict]:
    """Rebuild the frontier table from every length present on disk.

    Collected from the files rather than from this run, so a sweep over one
    length extends the table instead of replacing it.
    """
    lengths = sorted({int(p.stem.split("_L")[-1])
                      for p in OUT.glob("net_L*.json")}
                     | {int(p.stem.split("_L")[-1])
                        for p in OUT.glob("gmm_L*.json")})
    rows = [collect(length) for length in lengths]
    (OUT / "frontier.json").write_text(
        json.dumps({"n_train": N_TRAIN, "reps": reps, "rows": rows}, indent=2) + "\n")
    return rows


def collect(length: int) -> dict:
    """Headline numbers for one length, computed from the cells themselves.

    Two quantities per estimator. The root mean squared error of H pooled over
    intermittency at H = 0.10, which the pre-registered threshold concerns. And
    the error of lambda squared relative to its true value, pooled over the
    cells with H <= 0.2, which is the region the indices occupy and the
    quantity the test of monofractality depends on.
    """
    row: dict = {"length": length}
    sources = (("net", OUT / f"net_L{length}.json", ""),
               ("gmm", OUT / f"gmm_L{length}.json", "gmm_"))
    for name, path, prefix in sources:
        if not path.exists():
            continue
        rows = json.loads(path.read_text())["rows"]
        row[f"{name}_H_rmse_at_0.10"] = _pool(
            rows, f"{prefix}H_rmse", lambda r: abs(r["H_true"] - 0.10) < 1e-9)
        rel = [r[f"{prefix}lambda_rmse"] / r["lambda_sq_true"]
               for r in rows if r["H_true"] <= 0.2 + 1e-9]
        row[f"{name}_lambda_rel"] = float(np.mean(rel)) if rel else None
        row[f"{name}_cells"] = len(rows)
    return row


if __name__ == "__main__":
    main()
