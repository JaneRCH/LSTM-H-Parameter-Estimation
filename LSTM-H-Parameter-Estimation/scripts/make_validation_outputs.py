#!/usr/bin/env python3
"""Emit the validation tables and macros for the interim report.

Numbers are never transcribed by hand: this script writes the LaTeX fragments
that the report inputs, so the document and the measurements cannot drift.

Usage
-----
    PYTHONPATH=src python3 scripts/make_validation_outputs.py
"""
from __future__ import annotations

import json
import pathlib
import platform
import subprocess
import sys

import numpy as np

from rvmf.seeds import MASTER_SEED, seed_table
from rvmf.sfbm import CUTOFF_RATIO

REPORT = pathlib.Path("outputs")
TABLES = REPORT / "tables"
RESULTS = pathlib.Path("results_v2")

# Lag columns shown in the structure-function table. The full grid is wider; a
# report table that needs scrolling is a report table nobody reads.
SHOWN_LAGS = [1, 8, 64, 512]


def _fmt_pct(x: float) -> str:
    return f"${100 * x:+.2f}$"


def structure_table() -> str:
    d = json.loads((RESULTS / "validation.json").read_text())
    lags = d["lags"]
    idx = [lags.index(L) for L in SHOWN_LAGS]
    # The clipped-eigenvalue share is at most 3e-7 across the grid, so a column
    # of it would be a column of zeros that pushes the table past the margin.
    # The worst case goes in the caption via \valDefect instead.
    head = " & ".join(rf"$\tau={L}$" for L in SHOWN_LAGS)
    lines = [
        r"\begin{tabular}{@{}S[table-format=1.2] S[table-format=1.3] "
        + "r" * len(SHOWN_LAGS)
        + r"@{}}",
        r"\toprule",
        r"& & \multicolumn{%d}{c}{Relative error, per cent} \\" % len(SHOWN_LAGS),
        r"\cmidrule(lr){3-%d}" % (2 + len(SHOWN_LAGS)),
        r"{$H$} & {$\lambda^2$} & " + head + r" \\",
        r"\midrule",
    ]
    for r in d["rows"]:
        rel = [_fmt_pct(r["rel"][i]) for i in idx]
        lines.append(f"{r['H']:.2f} & {r['lambda_sq']:.3f} & " + " & ".join(rel) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def seed_tex() -> str:
    t = seed_table()
    lines = [
        r"\begin{tabular}{@{}l S[table-format=10.0]@{}}",
        r"\toprule",
        r"{Stream} & {Derived seed} \\",
        r"\midrule",
    ]
    for name, value in t.items():
        safe = name.replace("_", r"\_")
        lines.append(rf"\texttt{{{safe}}} & {value} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def environment_tex() -> str:
    def version(mod: str) -> str:
        try:
            return __import__(mod).__version__
        except Exception:
            return "not installed"

    rows = [
        ("Python", platform.python_version()),
        ("NumPy", version("numpy")),
        ("SciPy", version("scipy")),
        ("PyTorch", version("torch")),
        ("pandas", version("pandas")),
        ("Matplotlib", version("matplotlib")),
    ]
    lines = [
        r"\begin{tabular}{@{}l l@{}}",
        r"\toprule",
        r"{Component} & {Version} \\",
        r"\midrule",
    ]
    for name, ver in rows:
        lines.append(rf"{name} & \texttt{{{ver}}} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def macros() -> str:
    d = json.loads((RESULTS / "validation.json").read_text())
    rows = d["rows"]
    lag1 = max(abs(r["rel"][d["lags"].index(1)]) for r in rows)
    worst = max(r["max_abs_rel"] for r in rows)
    defect = max(max(r["defect"], 0.0) for r in rows)
    kurt_lo = min(r["kurtosis"] for r in rows)
    kurt_hi = max(r["kurtosis"] for r in rows)
    try:
        tests = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-q", "--no-header"],
            capture_output=True, text=True, env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"},
        ).stdout
        n_tests = next(
            (w for line in tests.splitlines() if "passed" in line for w in line.split()
             if w.isdigit()), "?"
        )
    except Exception:
        n_tests = "?"
    return "\n".join([
        rf"\newcommand{{\valReps}}{{{d['reps']}}}",
        rf"\newcommand{{\valLength}}{{{d['n']:,}}}".replace(",", r"{,}"),
        rf"\newcommand{{\valLagOne}}{{{100 * lag1:.2f}}}",
        rf"\newcommand{{\valWorst}}{{{100 * worst:.2f}}}",
        rf"\newcommand{{\valDefect}}{{{defect:.1e}}}",
        rf"\newcommand{{\valKurtLo}}{{{kurt_lo:.3f}}}",
        rf"\newcommand{{\valKurtHi}}{{{kurt_hi:.3f}}}",
        rf"\newcommand{{\valTests}}{{{n_tests}}}",
        rf"\newcommand{{\masterSeed}}{{{MASTER_SEED}}}",
        rf"\newcommand{{\cutoffRatio}}{{{CUTOFF_RATIO:.4f}}}",
    ])


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    written = {
        "validation_structure.tex": structure_table(),
        "validation_seeds.tex": seed_tex(),
        "validation_environment.tex": environment_tex(),
        "macros_validation.tex": macros(),
    }
    for name, body in written.items():
        (TABLES / name).write_text(body + "\n")
        print(f"wrote tables/{name}")


if __name__ == "__main__":
    main()
