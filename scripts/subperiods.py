#!/usr/bin/env python3
"""RQ5: does the estimated position differ across sub-periods?

The design follows from Section 5.5 rather than from the episodes one would
like to study. The window that identifies H is longer than a crisis, and the
test of monofractality keeps its power at a quarter of that length. So the two
parameters are asked at two resolutions:

``lambda^2``
    Consecutive non-overlapping blocks of ``SHORT`` trading days, the length at
    which the test retains power 0.93 against the weakest intermittency the
    indices carry. Each block is tested against a monofractal null simulated at
    that block's own estimated H and at the block's length.

``H``
    Consecutive non-overlapping blocks of ``LONG`` trading days, the shortest
    window at which the pre-registered threshold is met.

Blocks are laid down mechanically from the start of each index's sample. They
are not chosen to coincide with episodes, because boundaries chosen after
seeing the series are boundaries fitted to the answer. Each block is labelled
afterwards by the dates it happens to span.

Run:
    PYTHONPATH=src python3 scripts/subperiods.py --reps 200
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

from rvmf.corpus_v2 import PriorSpec, draw_parameters
from rvmf.data import clean_ohlc, impose_stale_open, load_ohlc, stale_open_share
from rvmf.features_v2 import FeatureSpecV2, extract
from rvmf.model_v2 import ModelConfigV2, ThetaNet, predict_theta
from rvmf.observation import simulate_bars
from rvmf.seeds import generator
from rvmf.sfbm import SfbmParams

SHORT: int = 250
LONG: int = 1000

#: Identical to scripts/empirical_v2.py and scripts/power_lambda.py.
LAMBDA_NULL: float = 1e-4

CODES: tuple[str, ...] = ("J200", "J203", "J210", "J213")
BAR_KEYS = ("open", "high", "low", "close")


def _replicate_index(code: str, modulus: int = 997) -> int:
    """Deterministic sub-stream index for an index code."""
    return int.from_bytes(hashlib.sha256(code.encode("utf-8")).digest()[:4], "big") % modulus


def load_net(path: Path) -> ThetaNet:
    ckpt = torch.load(path, weights_only=False)
    net = ThetaNet(ModelConfigV2(**ckpt["config"]))
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net


def index_bars(path: Path) -> tuple[dict[str, np.ndarray], np.ndarray, dict]:
    """Cleaned log bars under the simulator's recording convention."""
    frame, rep = clean_ohlc(load_ohlc(path))
    before = stale_open_share(frame)
    frame = impose_stale_open(frame)
    bars = {k: np.log(frame[k].to_numpy(dtype=float)) for k in BAR_KEYS}
    meta = {"rows": len(frame), "stale_open_share_before": before,
            "contaminated_prints": rep.contaminated_prints}
    return bars, frame.index.to_numpy(), meta


def estimate_block(bars: dict[str, np.ndarray], lo: int, hi: int,
                   spec: FeatureSpecV2, net: ThetaNet) -> tuple[float, float]:
    block = {k: v[lo:hi] for k, v in bars.items()}
    seq, summ = extract(block, spec)
    theta = predict_theta(net, seq[None, ...], summ[None, ...])[0]
    return float(theta[0]), float(theta[1])


def null_p_value(H: float, length: int, observed: float, reps: int,
                 prior: PriorSpec, spec: FeatureSpecV2, net: ThetaNet,
                 rng: np.random.Generator) -> float:
    """Share of monofractal replicates whose estimate reaches the observed one.

    The null is simulated at the block's own estimated H, so the comparison is
    with a non-multifractal process of the same roughness and the same length
    rather than with a single reference process.
    """
    draws = np.empty(reps)
    for i in range(reps):
        _, obs = draw_parameters(length, prior, rng)
        sf = SfbmParams(H=max(H, 0.0), lambda_sq=LAMBDA_NULL,
                        T=float(np.exp(rng.uniform(*np.log(prior.T_over_L)))) * length,
                        sigma_bar=float(rng.uniform(*prior.sigma_bar)))
        seq, summ = extract(simulate_bars(length, sf, obs, rng), spec)
        draws[i] = predict_theta(net, seq[None, ...], summ[None, ...])[0][1]
    return float(np.mean(draws >= observed))


def dispersion_test(
    estimates: list[float], which: int, H0: float, lam0: float, length: int,
    sets: int, prior: PriorSpec, spec: FeatureSpecV2, net: ThetaNet,
    rng: np.random.Generator,
) -> dict:
    """Is the spread across blocks larger than a constant process produces?

    A range of estimates across sub-periods is not evidence of time variation on
    its own, because each estimate carries error and short blocks carry a lot of
    it. The null here is the index's own full-sample position held fixed: as
    many blocks as the index has, each of the same length, simulated at
    ``(H0, lam0)``. The statistic is the standard deviation across a set of
    blocks, and the p-value is the share of simulated sets whose spread reaches
    the observed one.

    ``which`` selects the parameter, 0 for H and 1 for lambda squared.
    """
    observed = float(np.std(estimates, ddof=1))
    blocks = len(estimates)
    null = np.empty(sets)
    for s in range(sets):
        draws = np.empty(blocks)
        for b in range(blocks):
            _, obs = draw_parameters(length, prior, rng)
            sf = SfbmParams(
                H=max(H0, 0.0), lambda_sq=max(lam0, LAMBDA_NULL),
                T=float(np.exp(rng.uniform(*np.log(prior.T_over_L)))) * length,
                sigma_bar=float(rng.uniform(*prior.sigma_bar)))
            seq, summ = extract(simulate_bars(length, sf, obs, rng), spec)
            draws[b] = predict_theta(net, seq[None, ...], summ[None, ...])[0][which]
        null[s] = np.std(draws, ddof=1)
    return {
        "observed_sd": observed,
        "null_sd_median": float(np.median(null)),
        "null_sd_q95": float(np.quantile(null, 0.95)),
        "p_value": float(np.mean(null >= observed)),
        "blocks": blocks, "sets": sets,
    }


#: Midpoints of the stress episodes the sample spans, stated here rather than
#: chosen after inspecting the estimates. A block counts as a stress block when
#: it contains one of these dates, which is a rule a block either meets or does
#: not, unlike a rule based on how much of an episode it happens to cover.
STRESS_MIDPOINTS: tuple[str, ...] = (
    "2009-02-15",   # global financial crisis
    "2016-01-15",   # commodity collapse and domestic stress
    "2020-09-15",   # pandemic
    "2022-07-15",   # invasion of Ukraine
)


def stress_contrast(blocks: list[dict], draws: int, rng: np.random.Generator) -> dict:
    """Is the intermittency higher in blocks covering a stress episode?

    A permutation test on the block estimates. The statistic is the difference
    in mean intermittency between blocks that contain a stress midpoint and
    those that do not, and the reference distribution comes from reassigning
    the labels at random. With four episodes the test detects only a large
    contrast, which is reported alongside the result rather than left implicit.
    """
    stress = np.array([
        any(b["start"] <= m <= b["end"] for m in STRESS_MIDPOINTS) for b in blocks])
    lam = np.array([b["lambda_sq"] for b in blocks])
    if stress.sum() == 0 or stress.all():
        return {"stress_blocks": int(stress.sum()), "p_value": None}
    observed = float(lam[stress].mean() - lam[~stress].mean())
    null = np.empty(draws)
    for i in range(draws):
        perm = rng.permutation(stress)
        null[i] = lam[perm].mean() - lam[~perm].mean()
    return {
        "stress_blocks": int(stress.sum()),
        "calm_blocks": int((~stress).sum()),
        "mean_stress": float(lam[stress].mean()),
        "mean_calm": float(lam[~stress].mean()),
        "difference": observed,
        "p_value": float(np.mean(np.abs(null) >= abs(observed))),
        "draws": draws,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--codes", nargs="*", default=list(CODES))
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--net-dir", default="results_v2/net")
    ap.add_argument("--sets", type=int, default=250,
                    help="simulated sets of blocks for the dispersion test")
    ap.add_argument("--out", default="results_v2/subperiods.json")
    args = ap.parse_args()

    prior, spec = PriorSpec(), FeatureSpecV2()
    short_net = load_net(Path(args.net_dir) / f"theta_pw{SHORT}.pt")
    long_net = load_net(Path(args.net_dir) / f"theta_pw{LONG}.pt")
    t0 = time.time()
    out: dict = {"short": SHORT, "long": LONG, "reps": args.reps,
                 "lambda_null": LAMBDA_NULL, "indices": {}}

    for code in args.codes:
        bars, dates, meta = index_bars(
            Path(args.data_dir) / f"OHLC_historical_data_{code}.csv")
        n = len(dates)
        # A stable replicate index per index code. Python's hash() is salted
        # per process, so it cannot be used where the study must reproduce.
        rng = generator("episodes", replicate=_replicate_index(code))
        entry = {**meta, "lambda_blocks": [], "H_blocks": []}

        for start in range(0, n - SHORT + 1, SHORT):
            H, lam = estimate_block(bars, start, start + SHORT, spec, short_net)
            p = null_p_value(H, SHORT, lam, args.reps, prior, spec, short_net, rng)
            entry["lambda_blocks"].append({
                "start": str(dates[start])[:10],
                "end": str(dates[start + SHORT - 1])[:10],
                "H": H, "lambda_sq": lam, "p_value": p,
            })

        for start in range(0, n - LONG + 1, LONG):
            H, lam = estimate_block(bars, start, start + LONG, spec, long_net)
            entry["H_blocks"].append({
                "start": str(dates[start])[:10],
                "end": str(dates[start + LONG - 1])[:10],
                "H": H, "lambda_sq": lam,
            })

        # Whether the position moves at all, against a constant-position null
        # placed at the index's own pooled estimate.
        lam_hat = float(np.median([b["lambda_sq"] for b in entry["lambda_blocks"]]))
        H_hat = float(np.median([b["H"] for b in entry["H_blocks"]]))
        entry["lambda_dispersion"] = dispersion_test(
            [b["lambda_sq"] for b in entry["lambda_blocks"]], 1, H_hat, lam_hat,
            SHORT, args.sets, prior, spec, short_net, rng)
        entry["H_dispersion"] = dispersion_test(
            [b["H"] for b in entry["H_blocks"]], 0, H_hat, lam_hat,
            LONG, args.sets, prior, spec, long_net, rng)

        entry["stress_contrast"] = stress_contrast(
            entry["lambda_blocks"], 20000, rng)

        out["indices"][code] = entry
        lams = [b["lambda_sq"] for b in entry["lambda_blocks"]]
        hs = [b["H"] for b in entry["H_blocks"]]
        rejected = sum(b["p_value"] < 0.05 for b in entry["lambda_blocks"])
        ld, hd, sc = (entry["lambda_dispersion"], entry["H_dispersion"],
                      entry["stress_contrast"])
        print(f"{code}: {len(lams)} short blocks, lambda^2 {min(lams):.3f} to "
              f"{max(lams):.3f}, {rejected}/{len(lams)} reject monofractality, "
              f"spread {ld['observed_sd']:.4f} vs null {ld['null_sd_median']:.4f} "
              f"p={ld['p_value']:.3f} | {len(hs)} long blocks, H {min(hs):.3f} to "
              f"{max(hs):.3f}, spread {hd['observed_sd']:.4f} vs null "
              f"{hd['null_sd_median']:.4f} p={hd['p_value']:.3f}\n"
              f"      stress {sc['stress_blocks']} blocks mean "
              f"{sc.get('mean_stress', float('nan')):.4f} vs calm "
              f"{sc.get('mean_calm', float('nan')):.4f} p={sc['p_value']}",
              flush=True)

    out["wall_clock_min"] = (time.time() - t0) / 60.0
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
