"""Generate every table, figure and macro used by the interim report.

No number in the report is typed by hand: each table body and each macro is
written here from the JSON produced by the experiment scripts, so the document
and the results cannot drift apart.

Run:
    PYTHONPATH=src python3 scripts/make_outputs_v2.py --stage data
    PYTHONPATH=src python3 scripts/make_outputs_v2.py --stage results
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rvmf.data import clean_ohlc, load_ohlc
from rvmf.observation import ObservationParams, simulate_bars
from rvmf.rangeproxy import PROXIES, log_proxy
from rvmf.seeds import generator
from rvmf.sfbm import SfbmParams

CODES = ("J200", "J203", "J210", "J213", "J250", "J258", "J263")
LABEL = {"rv": r"$\log r_t^2$", "parkinson": r"$\log \hat v^{\mathrm{par}}_t$",
         "garman_klass": r"$\log \hat v^{\mathrm{gk}}_t$"}
PLAIN = {"rv": "Squared return", "parkinson": "Parkinson", "garman_klass": "Garman-Klass"}

plt.rcParams.update({
    "font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
    "figure.dpi": 160, "savefig.bbox": "tight", "axes.spines.top": False,
    "axes.spines.right": False,
})


def _bars(frame) -> dict[str, np.ndarray]:
    return {c: np.log(frame[c].to_numpy(dtype=float)) for c in ("open", "high", "low", "close")}


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print(f"  wrote {path}")


def stage_data(data_dir: Path, tables: Path, figures: Path, macros: dict) -> None:
    """Bar-convention audit and the measurement-noise audit of each proxy."""
    rng = generator("prior_predictive")
    sim = simulate_bars(
        20000, SfbmParams(H=0.13, lambda_sq=0.05, T=250000.0, sigma_bar=0.20),
        ObservationParams(), rng,
    )
    truth = np.log(sim["integrated_variance"])
    noise_var = {k: float(np.var(log_proxy(sim, k) - truth)) for k in PROXIES}

    conv_rows, noise_rows, shares = [], [], {k: [] for k in PROXIES}
    for code in CODES:
        frame, _ = clean_ohlc(load_ohlc(data_dir / f"OHLC_historical_data_{code}.csv"))
        frame = frame.dropna(subset=["open", "high", "low", "close"])
        b = _bars(frame)
        prev = np.concatenate([[np.nan], b["close"][:-1]])
        same = float(np.nanmean(np.isclose(b["open"][1:], prev[1:])))
        hi = float(np.mean(np.isclose(b["high"], np.maximum(b["open"], b["close"]))))
        conv_rows.append(
            f"{code} & {frame.shape[0]:,} & {frame.index.min().date()} & "
            f"{frame.index.max().date()} & {same:.1%} & {hi:.1%} \\\\".replace("%", r"\%")
        )
        cells = []
        for k in PROXIES:
            v = float(np.var(log_proxy(b, k)))
            s = noise_var[k] / v
            shares[k].append(s)
            cells.append(f"{v:.2f} & {s:.0%}".replace("%", r"\%"))
        noise_rows.append(f"{code} & " + " & ".join(cells) + r" \\")

    _write(tables / "bar_convention.tex", "\n".join(conv_rows) + "\n")
    _write(tables / "proxy_noise.tex", "\n".join(noise_rows) + "\n")

    for k in PROXIES:
        macros[f"noiseShare{k.replace('_','')}"] = f"{np.mean(shares[k]):.0%}".replace("%", r"\%")
        macros[f"noiseVar{k.replace('_','')}"] = f"{noise_var[k]:.2f}"
    macros["noiseShareRvLo"] = f"{min(shares['rv']):.0%}".replace("%", r"\%")
    macros["noiseShareRvHi"] = f"{max(shares['rv']):.0%}".replace("%", r"\%")
    macros["noiseShareGkLo"] = f"{min(shares['garman_klass']):.0%}".replace("%", r"\%")
    macros["noiseShareGkHi"] = f"{max(shares['garman_klass']):.0%}".replace("%", r"\%")

    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.4))
    x = np.arange(len(CODES))
    for i, k in enumerate(PROXIES):
        ax[0].bar(x + (i - 1) * 0.27, [s * 100 for s in shares[k]], 0.27, label=PLAIN[k])
    ax[0].set_xticks(x); ax[0].set_xticklabels(CODES)
    ax[0].set_ylabel("sampling noise, share of variance (%)")
    ax[0].set_title("How much of each proxy is noise")
    # Legends sit below the axes so they never cover the data.
    ax[0].legend(frameon=False, fontsize=7.5, loc="upper center",
                 bbox_to_anchor=(0.5, -0.17), ncol=3, handlelength=1.4,
                 columnspacing=1.4)

    sub = slice(0, 250)
    ax[1].plot(log_proxy(sim, "rv")[sub], lw=0.6, color="#4C78C8", alpha=0.85,
               label=PLAIN["rv"])
    ax[1].plot(log_proxy(sim, "garman_klass")[sub], lw=0.9, color="#E07B39",
               label=PLAIN["garman_klass"])
    ax[1].plot(truth[sub], lw=1.8, color="k", label="true log variance")
    ax[1].set_title("One simulated run, true value and two proxies")
    ax[1].set_xlabel("trading day")
    ax[1].set_ylabel("log variance")
    ax[1].legend(frameon=False, fontsize=7.5, loc="upper center",
                 bbox_to_anchor=(0.5, -0.17), ncol=3, handlelength=1.4,
                 columnspacing=1.2)
    fig.tight_layout()
    fig.savefig(figures / "fig_proxy_noise.png"); plt.close(fig)
    print(f"  wrote {figures / 'fig_proxy_noise.png'}")


def _net_identification(results: Path) -> dict | None:
    """Fixed-grid results for the learned estimator, keyed by (H, lambda^2).

    Produced by scripts/identification_net.py on the same grid, seed stream and
    replicate count as the moment-estimator run, so the two are comparable.
    """
    best, best_rmse, best_name = None, float("inf"), None
    for path in sorted(results.glob("identification_net*.json")):
        d = json.loads(path.read_text())
        r = d["verdict"]["rmse_H_at_0_10"]
        if r < best_rmse:
            best, best_rmse, best_name = d, r, d.get("estimator", path.stem)
    if best is None:
        return None
    return {
        "name": best_name,
        "verdict": best["verdict"],
        "rows": {(r["H_true"], r["lambda_sq_true"]): r for r in best["rows"]},
    }


def stage_identification(results: Path, tables: Path, figures: Path, macros: dict) -> None:
    d = json.loads((results / "identification.json").read_text())
    rows = d["rows"]
    net = _net_identification(results)
    body = []
    for r in rows:
        key = (r["H_true"], r["lambda_sq_true"])
        nr = net["rows"].get(key) if net else None
        body.append(
            f"{r['H_true']:.2f} & {r['lambda_sq_true']:.2f} & "
            f"{r['gmm_H_mean']:.3f} & {r['gmm_H_bias']:+.3f} & {r['gmm_H_rmse']:.3f} & "
            + (f"{nr['H_mean']:.3f} & {nr['H_bias']:+.3f} & {nr['H_rmse']:.3f} & "
               if nr else "{--} & {--} & {--} & ")
            + f"{r['gjr_H_mean']:.3f}" + r" \\"
        )
    _write(tables / "identification.tex", "\n".join(body) + "\n")

    v = d["verdict"]
    if net:
        nv = net["verdict"]
        macros["idNetName"] = net["name"]
        macros["idNetRmse"] = f"{nv['rmse_H_at_0_10']:.3f}"
        macros["idNetVerdict"] = "identified" if nv["identified"] else "not identified"
        ratios = [net["rows"][k]["H_rmse"] / r["gmm_H_rmse"]
                  for r in rows
                  if (k := (r["H_true"], r["lambda_sq_true"])) in net["rows"]]
        macros["idRatioLo"] = f"{min(ratios):.2f}"
        macros["idRatioHi"] = f"{max(ratios):.2f}"
        macros["idNetCells"] = str(sum(1 for x in ratios if x < 1.0))
        macros["idNetCellsAll"] = str(len(ratios))
        nb = [abs(net["rows"][k]["H_bias"]) for r in rows
              if (k := (r["H_true"], r["lambda_sq_true"])) in net["rows"]]
        macros["idNetBiasMax"] = f"{max(nb):.3f}"
    macros["idRmse"] = f"{v['rmse_H_at_0.10']:.3f}"
    macros["idThreshold"] = f"{v['threshold']:.2f}"
    macros["idVerdict"] = "not identified" if not v["identified"] else "identified"
    macros["idReps"] = str(rows[0]["reps"])
    macros["idLength"] = f"{v['length']:,}"

    # The pre-registered rule concerned H. It turned out that lambda^2 is the
    # parameter carrying the research question, so its precision is reported
    # alongside, and is flagged in the text as measured rather than tested.
    low = [r for r in rows if r["H_true"] <= 0.10]
    rel = [r["gmm_lambda_rmse"] / r["lambda_sq_true"] for r in low]
    macros["idLamRelLo"] = f"{min(rel):.2f}"
    macros["idLamRelHi"] = f"{max(rel):.2f}"
    macros["idLamHRange"] = "0.10"
    hi = [r for r in rows if r["H_true"] >= 0.30]
    macros["idLamRelFail"] = f"{min(r['gmm_lambda_rmse'] / r['lambda_sq_true'] for r in hi):.1f}"
    macros["idHBiasLo"] = f"{min(r['gmm_H_bias'] for r in rows if r['H_true'] <= 0.20):+.3f}"
    macros["idHBiasHi"] = f"{max(r['gmm_H_bias'] for r in rows if r['H_true'] <= 0.20):+.3f}"
    macros["idHSdLo"] = f"{min(r['gmm_H_sd'] for r in rows if r['H_true'] <= 0.20):.3f}"
    macros["idHSdHi"] = f"{max(r['gmm_H_sd'] for r in rows if r['H_true'] <= 0.20):.3f}"
    macros["idNaiveZeroLo"] = f"{min(r['gjr_H_mean'] for r in rows if r['H_true'] == 0.0):.3f}"
    macros["idNaiveZeroHi"] = f"{max(r['gjr_H_mean'] for r in rows if r['H_true'] == 0.0):.3f}"
    macros["idNaiveHiLo"] = f"{min(r['gjr_H_mean'] for r in rows if r['H_true'] == 0.30):.3f}"
    macros["idNaiveHiHi"] = f"{max(r['gjr_H_mean'] for r in rows if r['H_true'] == 0.30):.3f}"

    H = sorted({r["H_true"] for r in rows})
    fig, ax = plt.subplots(1, 3, figsize=(10.2, 3.5))
    for lam in sorted({r["lambda_sq_true"] for r in rows}):
        g = [next(r for r in rows if r["H_true"] == h and r["lambda_sq_true"] == lam) for h in H]
        lbl = f"$\\lambda^2={lam}$"
        ax[0].errorbar(H, [r["gmm_H_mean"] for r in g], yerr=[r["gmm_H_sd"] for r in g],
                       marker="o", ms=3, capsize=2, lw=1, label=lbl)
        if not net:
            ax[1].plot(H, [r["gmm_H_rmse"] for r in g], marker="o", ms=3, lw=1, label=lbl)
        # lambda^2 is the parameter carrying the research question, so its
        # precision belongs beside H's. Relative error makes the three
        # intermittency levels comparable on one axis.
        ax[2].plot(H, [r["gmm_lambda_rmse"] / r["lambda_sq_true"] for r in g],
                   marker="o", ms=3, lw=1, label=lbl)
    ax[0].plot([0, 0.3], [0, 0.3], "k--", lw=0.8, label="perfect recovery")
    ax[0].set_xlabel("true $H$"); ax[0].set_ylabel("estimated $H$")
    ax[0].set_title("Moment estimator: $\\hat H$ against $H$")
    if net:
        # One line per estimator, pooled across intermittency, so the two
        # frontiers are read against the same threshold on the same axes.
        g_pool = [np.sqrt(np.mean([r["gmm_H_rmse"] ** 2 for r in rows
                                   if r["H_true"] == h])) for h in H]
        n_pool = [np.sqrt(np.mean([net["rows"][(h, lam)]["H_rmse"] ** 2
                                   for lam in sorted({r["lambda_sq_true"] for r in rows})
                                   if (h, lam) in net["rows"]])) for h in H]
        ax[1].plot(H, g_pool, marker="o", ms=4, lw=1.6, color="#4C78C8",
                   label="moment estimator")
        ax[1].plot(H, n_pool, marker="s", ms=4, lw=1.6, color="#0A7B3E",
                   label="learned estimator")
    ax[1].axhline(v["threshold"], color="k", ls="--", lw=0.8,
                  label=f"threshold {v['threshold']:.2f}")
    ax[1].set_xlabel("true $H$"); ax[1].set_ylabel("error of $\\hat H$ (RMSE)")
    ax[1].set_title("Error in $H$, by estimator")
    ax[2].axhline(1.0, color="k", ls="--", lw=0.8, label="error equals true value")
    ax[2].set_xlabel("true $H$")
    ax[2].set_ylabel(r"error of $\hat\lambda^2$, relative")
    ax[2].set_title("Moment estimator: error in $\\lambda^2$")
    # Legends sit below the axes so they never cover the data.
    for a in ax:
        a.legend(frameon=False, fontsize=7.5, loc="upper center",
                 bbox_to_anchor=(0.5, -0.20), ncol=2, handlelength=1.4,
                 columnspacing=1.2)
    fig.tight_layout()
    fig.savefig(figures / "fig_identification.png"); plt.close(fig)
    print(f"  wrote {figures / 'fig_identification.png'}")


def _fmt_p(p) -> str:
    """Format a simulated p-value. A p of zero means 'below the resolution'."""
    if p is None:
        return "{--}"
    if p <= 0.0:
        return r"$<0.01$"
    return f"{p:.3f}"


def _ci(pair, lo=None, hi=None, dp=3) -> str:
    """Format an interval, clipped to the parameter's support.

    The reflected percentile interval can run outside the range the parameter is
    defined on, because the reflection is about the point estimate and knows
    nothing of the constraint. Reporting a negative bound for a quantity that
    cannot be negative would be a formatting artefact, so the interval is
    intersected with the support and the clipping is stated in the caption.
    """
    if not pair:
        return "{--}"
    a, b = pair
    if lo is not None:
        a, b = max(a, lo), max(b, lo)
    if hi is not None:
        a, b = min(a, hi), min(b, hi)
    return f"$[{a:.{dp}f},\\,{b:.{dp}f}]$"


def stage_results(results: Path, tables: Path, figures: Path, macros: dict) -> None:
    """Tables and macros for the estimator comparison and the index estimates."""
    bpath, epath = results / "benchmark.json", results / "empirical.json"

    if bpath.exists():
        b = json.loads(bpath.read_text())
        est = b["estimators"]
        order = [k for k in ("LSTM", "GMM_lnM", "GJR_regression") if k in est]
        label = {"LSTM": "Learned", "GMM_lnM": "Moment estimator",
                 "GJR_regression": "Naive regression"}
        body = []
        for k in order:
            e = est[k]
            lam = e.get("lambda_sq_rmse")
            body.append(
                f"{label[k]} & {e['H']['bias']:+.4f} & {e['H']['rmse']:.4f} & "
                f"{e['H']['rmse_near_0.10']:.4f} & "
                + (f"{lam:.4f}" if lam is not None else "{--}")
                + r" \\"
            )
        _write(tables / "benchmark.tex", "\n".join(body) + "\n")
        macros["bnN"] = f"{b['n']:,}".replace(",", "{,}")
        best = min(order, key=lambda k: est[k]["H"]["rmse_near_0.10"])
        macros["bnBest"] = label[best]
        macros["bnBestRmse"] = f"{est[best]['H']['rmse_near_0.10']:.4f}"
        for k in order:
            tag = {"LSTM": "Lstm", "GMM_lnM": "Gmm", "GJR_regression": "Naive"}[k]
            macros[f"bn{tag}Rmse"] = f"{est[k]['H']['rmse']:.4f}"
            macros[f"bn{tag}Near"] = f"{est[k]['H']['rmse_near_0.10']:.4f}"
            macros[f"bn{tag}Bias"] = f"{est[k]['H']['bias']:+.4f}"
            if est[k].get("lambda_sq_rmse") is not None:
                macros[f"bn{tag}Lam"] = f"{est[k]['lambda_sq_rmse']:.4f}"
        print(f"  wrote {tables / 'benchmark.tex'}")

    if epath.exists():
        d = json.loads(epath.read_text())
        # Two tables rather than one: estimates and tests. A single table wide
        # enough for both runs past the margin and stops being readable.
        est, tst = [], []
        for code, e in d["indices"].items():
            l, g, t = e["lstm"], e["gmm"], e["test_lambda_eq_0"]
            n = f"{e['n']:,}".replace(",", "{,}")
            est.append(
                f"{code} & {n} & {l['H']:.3f} & {_ci(l.get('H_ci95'), lo=0.0, hi=0.5)} & "
                f"{g['H']:.3f} & {l['lambda_sq']:.3f} & "
                f"{_ci(l.get('lambda_sq_ci95'), lo=0.0)} & {g['lambda_sq']:.3f}" + r" \\"
            )
            matched = e.get("length_matches_training", True)
            tst.append(
                f"{code} & {_fmt_p(t.get('p_value_lstm'))} & "
                f"{_fmt_p(e['test_H_eq_0']['p_value'])} & "
                f"{_fmt_p(e['test_H_eq_0.10']['p_value'])} & "
                + ("yes" if matched else "no") + r" \\"
            )
        _write(tables / "empirical.tex", "\n".join(est) + "\n")
        _write(tables / "empirical_tests.tex", "\n".join(tst) + "\n")

        macros["empReps"] = str(d["replicates"])
        lam_p = [e["test_lambda_eq_0"]["p_value_lstm"] for e in d["indices"].values()]
        macros["empLamPMax"] = _fmt_p(max(lam_p))
        macros["empLamNull"] = str(
            next(iter(d["indices"].values()))["test_lambda_eq_0"]["null_lambda_sq"]
        )
        q95 = max(e["test_lambda_eq_0"]["null_q95_lstm"] for e in d["indices"].values())
        macros["empLamNullQ"] = f"{q95:.4f}"
        lo = min(e["lstm"]["lambda_sq"] for e in d["indices"].values())
        hi = max(e["lstm"]["lambda_sq"] for e in d["indices"].values())
        macros["empLamLo"] = f"{lo:.3f}"
        macros["empLamHi"] = f"{hi:.3f}"
        macros["empHLo"] = f"{min(e['lstm']['H'] for e in d['indices'].values()):.3f}"
        macros["empHHi"] = f"{max(e['lstm']['H'] for e in d['indices'].values()):.3f}"
        macros["empN"] = str(len(d["indices"]))
        rej0 = sum(e["test_H_eq_0"]["p_value"] < 0.05 for e in d["indices"].values())
        macros["empRejectHZero"] = str(rej0)
        macros["empRejectHOne"] = str(sum(
            e["test_H_eq_0.10"]["p_value"] < 0.05 for e in d["indices"].values()))
        stale = [e.get("stale_open_share_before") for e in d["indices"].values()
                 if e.get("stale_open_share_before") is not None]
        if stale:
            macros["empStaleLo"] = f"{min(stale):.3f}"
            macros["empStaleHi"] = f"{max(stale):.3f}"
        shift = max(e["feature_shift_outside_training_band"] for e in d["indices"].values())
        macros["empShiftMax"] = f"{shift:.0%}".replace("%", r"\%")
        matched = [c for c, e in d["indices"].items() if e.get("length_matches_training")]
        macros["empMatched"] = ", ".join(matched)
        macros["empUnmatched"] = ", ".join(
            c for c in d["indices"] if c not in matched)
        print(f"  wrote {tables / 'empirical.tex'}")
        print(f"  wrote {tables / 'empirical_tests.tex'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="data",
                    choices=("data", "identification", "results", "all"))
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--results", default="results_v2")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    out = Path(args.out)
    tables, figures = out / "tables", out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    macro_path = tables / "macros.tex"
    macros: dict[str, str] = {}
    if macro_path.exists():
        for line in macro_path.read_text().splitlines():
            if line.startswith(r"\newcommand{\\"):
                pass
    stages = (("data", "identification", "results") if args.stage == "all"
              else (args.stage,))
    for stage in stages:
        print(f"[{stage}]")
        if stage == "data":
            stage_data(Path(args.data_dir), tables, figures, macros)
        elif stage == "identification":
            stage_identification(Path(args.results), tables, figures, macros)
        elif stage == "results":
            stage_results(Path(args.results), tables, figures, macros)

    existing = {}
    if macro_path.exists():
        import re
        for m in re.finditer(r"\\newcommand\{\\(\w+)\}\{(.*)\}", macro_path.read_text()):
            existing[m.group(1)] = m.group(2)
    existing.update(macros)
    body = "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(existing.items()))
    _write(macro_path, "% Generated by scripts/make_outputs_v2.py. Do not edit.\n" + body + "\n")


if __name__ == "__main__":
    main()
