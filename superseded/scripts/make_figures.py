"""Generate the report figures from the result JSON files.

Print-bound academic figures. Identity is never carried by colour alone: every
series also has its own line style and marker, so the figures survive greyscale
printing and colour-vision deficiency. Grid and axes are recessive, no figure
uses a second y-axis, and every figure has a counterpart table in the report.

Run:
    PYTHONPATH=src python3 scripts/make_figures.py
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

# Validated categorical palette (slots 1, 2, 3, 7 of the reference theme).
C = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "violet": "#4a3aa7",
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "grid": "#d8d8d4",
}
# Secondary encoding: each series gets a distinct dash and marker.
STYLE = [
    dict(color=C["blue"], linestyle="-", marker="o"),
    dict(color=C["orange"], linestyle="--", marker="s"),
    dict(color=C["aqua"], linestyle="-.", marker="^"),
    dict(color=C["violet"], linestyle=":", marker="D"),
]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.edgecolor": C["ink2"],
    "axes.linewidth": 0.7,
    "axes.grid": True,
    "grid.color": C["grid"],
    "grid.linewidth": 0.5,
    "grid.alpha": 0.9,
    "axes.axisbelow": True,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "legend.frameon": False,
})


def tidy(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return ax


def save(fig, out: Path, name: str):
    p = out / name
    fig.savefig(p)
    plt.close(fig)
    print(f"  wrote {name}")


# ---------------------------------------------------------------------------


def fig_series(out, data_dir):
    """J200 level and its log realised volatility, with crisis episodes shaded."""
    import pandas as pd

    from rvmf.data import clean_ohlc, load_ohlc
    from rvmf.proxies import Estimand, build_estimand

    frame, _ = clean_ohlc(load_ohlc(Path(data_dir) / "OHLC_historical_data_J200.csv"))
    x = build_estimand(frame, Estimand(proxy="rv", log=True, convention="series"))

    episodes = [
        ("2008-09-01", "2009-06-30", "2008 GFC"),
        ("2015-08-01", "2016-06-30", "2015-16 EM stress"),
        ("2020-02-15", "2021-03-31", "COVID-19"),
        ("2022-02-01", "2022-10-31", "2022 drawdown"),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(6.6, 4.4), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1]})
    for ax in axes:
        for lo, hi, _ in episodes:
            ax.axvspan(pd.Timestamp(lo), pd.Timestamp(hi), color=C["grid"],
                       alpha=0.55, lw=0)
        tidy(ax)

    axes[0].plot(frame.index, frame["close"], color=C["blue"], lw=1.1)
    axes[0].set_ylabel("Index level")
    axes[0].set_title("FTSE/JSE Top 40 (J200), 2006 to 2026", loc="left")

    axes[1].plot(x.index, x.to_numpy(), color=C["ink2"], lw=0.35)
    axes[1].set_ylabel(r"$\log|r_t|$")
    axes[1].set_xlabel("Date")

    for lo, hi, lab in episodes:
        mid = pd.Timestamp(lo) + (pd.Timestamp(hi) - pd.Timestamp(lo)) / 2
        axes[0].annotate(lab, xy=(mid, axes[0].get_ylim()[1]),
                         xytext=(0, 2), textcoords="offset points",
                         ha="center", va="bottom", fontsize=6.5, color=C["ink2"])
    save(fig, out, "fig_series.png")


def fig_jse_hq(out, emp):
    """Estimated h(q) for J200 under both estimators, with the anchored line."""
    from rvmf.ghe import anchored_fit

    q = np.array(emp["q"])
    v = emp["primary"]["rv-log-series"]
    h_mf = np.array(v["mfdfa"]["h"])
    se = np.array(v["mfdfa"]["stderr"])
    h_ls = np.array(v["lstm"]["h"])
    fit = anchored_fit(q, h_mf)

    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    tidy(ax)
    ax.fill_between(q, h_mf - 1.96 * se, h_mf + 1.96 * se,
                    color=C["blue"], alpha=0.15, lw=0)
    ax.plot(q, h_mf, **STYLE[0], lw=1.7, ms=4.5, label="MF-DFA")
    ax.plot(q, h_ls, **STYLE[1], lw=1.7, ms=4.5, label="LSTM")
    ax.plot(q, fit.predict(), color=C["ink2"], lw=1.0, linestyle=(0, (1, 2)),
            label=rf"anchored: $H={fit.H:.3f}$, $\lambda={fit.lam:.4f}$")
    ax.axhline(0.5, color=C["ink2"], lw=0.6, alpha=0.5)
    ax.annotate("$h=1/2$", xy=(q[0], 0.5), xytext=(2, 3),
                textcoords="offset points", ha="left", fontsize=7,
                color=C["ink2"])
    ax.set_xlabel("Moment order $q$")
    ax.set_ylabel("$h(q)$")
    ax.set_title("Generalised Hurst function, J200 log realised volatility",
                 loc="left")
    ax.legend(loc="upper right")
    save(fig, out, "fig_jse_hq.png")


def fig_jse_falpha(out, emp):
    """Singularity spectra under each estimator."""
    from rvmf.ghe import singularity_spectrum

    q = np.array(emp["q"])
    v = emp["primary"]["rv-log-series"]
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    tidy(ax)
    for i, (lab, key) in enumerate([("MF-DFA", "mfdfa"), ("LSTM", "lstm")]):
        s = singularity_spectrum(q, np.array(v[key]["h"]))
        ax.plot(s.alpha, s.f, **STYLE[i], lw=1.7, ms=4.5,
                label=rf"{lab}, $\Delta\alpha={s.width:.3f}$")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$f(\alpha)$")
    ax.set_title("Singularity spectrum, J200 log realised volatility", loc="left")
    ax.legend(loc="lower center")
    save(fig, out, "fig_jse_falpha.png")


def fig_nulls(out, emp):
    """Simulated null distributions of the width, with the observed value marked."""
    v = emp["primary"]["rv-log-series"]
    names = {
        "monofractal": "Monofractal (finite-size)",
        "heavy_tailed": "Rough heavy-tailed",
        "shuffle": "Distributional (shuffle)",
    }
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    for ax, (kind, lab) in zip(axes, names.items()):
        t = v["nulls"][kind]
        obs = t["observed_width"]
        draws = np.asarray(t.get("null_widths", []), dtype=float)
        tidy(ax)
        if draws.size:
            ax.hist(draws, bins=24, color=C["grid"], edgecolor=C["ink2"],
                    linewidth=0.4)
        ax.axvline(t["null_p95"], color=C["orange"], lw=1.3, linestyle="-.")
        ax.axvline(obs, color=C["blue"], lw=2.2)
        reject = "rejected" if t["p_value"] < 0.05 else "not rejected"
        ax.set_title(f"{lab}\n$p={t['p_value']:.3f}$ ({reject})",
                     loc="left", fontsize=8)
        ax.set_xlabel(r"$\Delta\alpha$")
        ax.tick_params(labelleft=False)
    axes[0].set_ylabel(f"replicates (n = {emp['n_null']})")
    handles = [
        plt.Line2D([], [], color=C["blue"], lw=2.2, label="observed width"),
        plt.Line2D([], [], color=C["orange"], lw=1.3, ls="-.",
                   label="null 95th percentile"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.13))
    fig.tight_layout()
    save(fig, out, "fig_nulls.png")


def fig_surrogates(out, emp):
    """Observed width against shuffled and IAAFT surrogates, per proxy."""
    labels, obs, shuf, iaaft = [], [], [], []
    pretty = {"rv-log-series": r"$\log|r_t|$",
              "yz-log-series-w22": "log Yang-Zhang"}
    for k, v in emp["primary"].items():
        s = v["surrogates"]
        labels.append(pretty.get(k, k))
        obs.append(s["observed_width"])
        shuf.append(s["shuffle_mean"])
        iaaft.append(s["iaaft_mean"])

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(5.6, 2.4))
    tidy(ax)
    h = 0.24
    ax.barh(y + h, obs, height=h, color=C["blue"], label="observed")
    ax.barh(y, iaaft, height=h, color=C["orange"], label="IAAFT surrogate")
    ax.barh(y - h, shuf, height=h, color=C["aqua"], label="shuffled surrogate")
    for yy, val in zip(y + h, obs):
        ax.annotate(f"{val:.3f}", xy=(val, yy), xytext=(3, 0),
                    textcoords="offset points", va="center", fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(r"spectrum width $\Delta\alpha$")
    ax.set_title("Surrogate decomposition of the observed width", loc="left")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    save(fig, out, "fig_surrogates.png")


def fig_benchmark(out, bench):
    """q-stratified RMSE of the two estimators on held-out synthetic paths."""
    q = np.array(bench["q"])
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    tidy(ax)
    ax.plot(q, bench["rmse_by_q"]["lstm"], **STYLE[0], lw=1.7, ms=4.5, label="LSTM")
    ax.plot(q, bench["rmse_by_q"]["mfdfa"], **STYLE[1], lw=1.7, ms=4.5, label="MF-DFA")
    ax.set_xlabel("Moment order $q$")
    ax.set_ylabel(r"$\mathrm{RMSE}(q)$")
    ax.set_title("Estimator error against known truth, held-out synthetic paths",
                 loc="left")
    ax.legend(loc="upper center")
    save(fig, out, "fig_benchmark.png")


def fig_fsize(out, val):
    """Width fabricated by MF-DFA on a process whose true width is zero."""
    rows = val["finite_size"]["rows"]
    n = np.array([r["n"] for r in rows], dtype=float)
    mean = np.array([r["mean_width"] for r in rows])
    sd = np.array([r["sd_width"] for r in rows])

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    tidy(ax)
    ax.fill_between(n, mean - sd, mean + sd, color=C["blue"], alpha=0.15, lw=0)
    ax.plot(n, mean, **STYLE[0], lw=1.7, ms=5)
    ax.axhline(0.0, color=C["ink2"], lw=0.8, linestyle="--")
    ax.annotate("true width = 0", xy=(n[0], 0), xytext=(2, 4),
                textcoords="offset points", fontsize=7.5, color=C["ink2"])
    ax.set_xscale("log")
    ax.set_xticks(n)
    ax.set_xticklabels([f"{int(v):,}".replace(",", " ") for v in n])
    ax.set_xlabel("Series length")
    ax.set_ylabel(r"fabricated $\Delta\alpha$")
    ax.set_title("MF-DFA fabricates width on a monofractal process", loc="left")
    save(fig, out, "fig_fsize.png")


def fig_tail(out, val):
    """The piecewise heavy-tailed label against what the process actually does."""
    rows = val["heavy_tail_label"]["rows"]
    qs = np.array([2.0, 3.0, 4.0, 5.0])
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8), sharey=False)
    for ax, r in zip(axes, [rows[0], rows[3]]):
        tidy(ax)
        ax.plot(qs, r["h_estimated_q2_to_q5"], **STYLE[0], lw=1.7, ms=5,
                label="estimated")
        ax.plot(qs, r["h_piecewise_q2_to_q5"], **STYLE[1], lw=1.7, ms=5,
                label="piecewise label")
        ax.axvline(r["gamma"], color=C["ink2"], lw=0.9, linestyle=":")
        ax.annotate(rf"$q=\gamma={r['gamma']:.1f}$", xy=(r["gamma"], ax.get_ylim()[1]),
                    xytext=(3, -8), textcoords="offset points", fontsize=7,
                    color=C["ink2"])
        ax.set_title(rf"$H={r['hurst']:.2f}$, $\gamma={r['gamma']:.1f}$", loc="left")
        ax.set_xlabel("Moment order $q$")
        ax.set_xticks(qs)
    axes[0].set_ylabel("$h(q)$")
    axes[0].legend(loc="lower left")
    save(fig, out, "fig_tail.png")



def fig_episodes(out, ep):
    """Sub-period profiles and the observed width against episode-specific nulls."""
    q = np.array(ep["q"])
    periods = ep["periods"]
    short = [p["label"].split(",")[0] for p in periods]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    for i, p in enumerate(periods):
        axes[0].plot(q, p["mfdfa"]["h"], **STYLE[i], lw=1.6, ms=4,
                     label=short[i])
    tidy(axes[0])
    axes[0].set_xlabel("Moment order $q$")
    axes[0].set_ylabel("$h(q)$")
    axes[0].set_title("Profile by sub-period", loc="left")
    axes[0].legend(loc="upper right", fontsize=7)

    ax = tidy(axes[1])
    y = np.arange(len(periods))
    for i, p in enumerate(periods):
        lo = p["null"]["null_mean"] - p["null"]["null_sd"]
        hi = p["null"]["null_mean"] + p["null"]["null_sd"]
        ax.plot([max(lo, 0), hi], [i, i], color=C["grid"], lw=6,
                solid_capstyle="butt",
                label="null mean $\\pm$ 1 s.d." if i == 0 else None)
        ax.plot(p["null"]["null_p95"], i, marker="|", ms=11,
                color=C["orange"], mew=1.6,
                label="null 95th pct." if i == 0 else None)
        ax.plot(p["mfdfa"]["width"], i, marker="o", ms=6, color=C["blue"],
                label="observed" if i == 0 else None)
        ax.annotate(f"$p={p['null']['p_value']:.3f}$",
                    xy=(p["mfdfa"]["width"], i), xytext=(8, 5),
                    textcoords="offset points", fontsize=7, color=C["ink2"])
    ax.set_yticks(y)
    ax.set_yticklabels(short, fontsize=7.5)
    ax.set_xlabel(r"spectrum width $\Delta\alpha$")
    ax.set_title("Width against each sub-period's own null", loc="left")
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.7, len(periods) - 0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3,
              fontsize=7, columnspacing=1.0, handletextpad=0.5)
    fig.tight_layout()
    save(fig, out, "fig_episodes.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="report/figures")
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    val = json.loads(Path("results/validation.json").read_text())
    emp = json.loads(Path("results/empirical.json").read_text())
    bench = json.loads(Path("results/benchmark_report.json").read_text())
    ep_path = Path("results/episodes.json")

    print("figures:")
    fig_series(out, args.data_dir)
    fig_jse_hq(out, emp)
    fig_jse_falpha(out, emp)
    fig_nulls(out, emp)
    fig_surrogates(out, emp)
    fig_benchmark(out, bench)
    fig_fsize(out, val)
    fig_tail(out, val)
    if ep_path.exists():
        fig_episodes(out, json.loads(ep_path.read_text()))


if __name__ == "__main__":
    main()
