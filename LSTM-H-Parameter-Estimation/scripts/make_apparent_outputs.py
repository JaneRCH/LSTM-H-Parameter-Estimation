#!/usr/bin/env python3
"""Emit the table and macros for the generalised-exponent correspondence.

Reads results_v2/apparent_hq.json and results_v2/gmm_only.json.

Usage
-----
    PYTHONPATH=src python3 scripts/make_apparent_outputs.py
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

REPORT = pathlib.Path("outputs")
TABLES = REPORT / "tables"
RESULTS = pathlib.Path("results_v2")

# One H per column block keeps the table readable; the pattern in lambda^2 is
# the point, and it is the same at every H.
SHOWN_H = 0.10
OBJECTS = (("omega", r"$\omega$"), ("measure", "Measure"), ("returns", "Returns"))


def table(rows: list[dict]) -> str:
    sel = sorted((r for r in rows if abs(r["H"] - SHOWN_H) < 1e-9),
                 key=lambda r: r["lambda_sq"])
    lines = [
        r"\begin{tabular}{@{}S[table-format=1.4] rr rr rr@{}}",
        r"\toprule",
        r"& \multicolumn{2}{c}{Log volatility $\omega$}"
        r" & \multicolumn{2}{c}{Integrated variance}"
        r" & \multicolumn{2}{c}{Returns} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"{$\lambda^2$} & $H(2)$ & $\mathrm{d}H/\mathrm{d}q$"
        r" & $H(2)$ & $\mathrm{d}H/\mathrm{d}q$"
        r" & $H(2)$ & $\mathrm{d}H/\mathrm{d}q$ \\",
        r"\midrule",
    ]
    for r in sel:
        cells = []
        for key, _ in OBJECTS:
            h2 = r[f"hq_{key}"][3]          # q = 2 is the fourth point of Q_GRID
            cells += [f"{h2:.3f}", f"${r['slope_' + key]:+.4f}$"]
        lines.append(f"{r['lambda_sq']:.4f} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def macros(rows: list[dict], fitted: dict) -> str:
    flat = max(abs(r["slope_omega"]) for r in rows)
    zero = [r for r in rows if r["lambda_sq"] < 1e-3]
    big = [r for r in rows if r["lambda_sq"] >= 0.0999]
    # Slope at the parameters the indices turn out to carry.
    lam_hat = float(np.median([v["lambda_sq"] for v in fitted.values()]))
    near = min(rows, key=lambda r: (abs(r["H"] - 0.10), abs(r["lambda_sq"] - lam_hat)))
    ret_lo = min(r["hq_returns"][3] for r in zero)
    ret_hi = max(r["hq_returns"][3] for r in big)
    return "\n".join([
        rf"\newcommand{{\hqFlat}}{{{flat:.4f}}}",
        rf"\newcommand{{\hqMeasZero}}{{{max(abs(r['slope_measure']) for r in zero):.4f}}}",
        rf"\newcommand{{\hqMeasBig}}{{{min(r['slope_measure'] for r in big):.3f}}}",
        rf"\newcommand{{\hqRetBig}}{{{min(r['slope_returns'] for r in big):.3f}}}",
        rf"\newcommand{{\hqAtFitMeas}}{{{near['slope_measure']:.3f}}}",
        rf"\newcommand{{\hqAtFitRet}}{{{near['slope_returns']:.3f}}}",
        rf"\newcommand{{\hqAtFitLam}}{{{near['lambda_sq']:.3f}}}",
        rf"\newcommand{{\hqPriceLo}}{{{ret_lo:.2f}}}",
        rf"\newcommand{{\hqPriceHi}}{{{ret_hi:.2f}}}",
    ])


def figure(rows: list[dict]) -> None:
    """Slope of H(q) against intermittency, one line per object.

    The point of the figure is that one of the three lines is flat and the other
    two are not, so the three are drawn on one axis at one scale.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 160, "savefig.bbox": "tight", "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.3,
    })
    sel = sorted((r for r in rows if abs(r["H"] - SHOWN_H) < 1e-9),
                 key=lambda r: r["lambda_sq"])
    lam = [r["lambda_sq"] for r in sel]
    styles = {
        "omega": ("Log volatility", "#4C78C8", "o", 2.0),
        "measure": ("Integrated variance", "#E07B39", "s", 1.4),
        "returns": ("Returns", "#1F6B45", "^", 1.4),
    }
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for key, (label, colour, marker, lw) in styles.items():
        ax.plot(lam, [r[f"slope_{key}"] for r in sel], marker=marker, ms=4,
                lw=lw, color=colour, label=label)
    ax.axhline(0.0, color="k", lw=0.8, ls="--", label="no multifractality")
    ax.set_xlabel(r"intermittency $\lambda^2$")
    ax.set_ylabel(r"slope of $H(q)$ in $q$")
    ax.set_title("Which series shows multifractality")
    # Legend below the axes so it never covers the data.
    ax.legend(frameon=False, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), ncol=2, handlelength=1.6)
    fig.tight_layout()
    out = REPORT / "figures" / "fig_apparent_hq.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    d = json.loads((RESULTS / "apparent_hq.json").read_text())
    fitted = json.loads((RESULTS / "gmm_only.json").read_text())
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "apparent_hq.tex").write_text(table(d["rows"]) + "\n")
    body = macros(d["rows"], fitted)
    body += f"\n\\newcommand{{\\hqReps}}{{{d['reps']}}}"
    body += f"\n\\newcommand{{\\hqScaleLo}}{{{min(d['scales'])}}}"
    body += f"\n\\newcommand{{\\hqScaleHi}}{{{max(d['scales'])}}}"
    (TABLES / "macros_apparent.tex").write_text(body + "\n")
    print("wrote tables/apparent_hq.tex")
    print("wrote tables/macros_apparent.tex")
    figure(d["rows"])


if __name__ == "__main__":
    main()
