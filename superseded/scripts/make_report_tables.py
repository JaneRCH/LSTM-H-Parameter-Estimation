"""Generate LaTeX table bodies and result macros from the result JSON files.

Every number in Section 4 of the report is produced here, so the report cannot
drift from the results it cites. Table structure (caption, label, column
specification) stays in the .tex; only the rows are generated.

Run:
    PYTHONPATH=src python3 scripts/make_report_tables.py \
        --validation results/validation.json \
        --train results/train_report.json \
        --out report/tables
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def f(x, p=3):
    """Format a number, or a visible placeholder if it is missing."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return r"\TBD{}"
    return f"{x:.{p}f}"


def write(path: Path, rows: list[str]) -> None:
    path.write_text("\n".join(rows) + "\n")
    print(f"  wrote {path.name} ({len(rows)} rows)")


# --------------------------------------------------------------------------


def simulator_labels(v, out):
    pretty = {
        "fbm_H0.30": r"fBm, $H=0.30$",
        "fbm_H0.70": r"fBm, $H=0.70$",
        "binomial_p0.60": r"Binomial, $p=0.60$",
        "binomial_p0.70": r"Binomial, $p=0.70$",
        "mmar_Hb0.30_lc0.40": r"MMAR, $H_b=0.30$, $\lambda_c^2=0.40$",
        "mmar_Hb0.25_lc0.40": r"MMAR, $H_b=0.25$, $\lambda_c^2=0.40$",
        "mmar_Hb0.40_lc0.60": r"MMAR, $H_b=0.40$, $\lambda_c^2=0.60$",
    }
    rows = [
        f"{pretty.get(r['case'], r['case'])} & {f(r['mae'],4)} & "
        f"{f(r['max_abs_error'],4)} & {f(r['h2_estimated'])} & "
        f"{f(r['h2_label'])} \\\\"
        for r in v["simulator_labels"]["rows"]
    ]
    write(out / "simval.tex", rows)


def anchored(v, out):
    rows = [
        f"$H_b={r['hurst_base']:.2f}$, $\\lambda_c^2={r['lambda_c2']:.2f}$ & "
        f"{f(r['H_true'])} & {f(r['H_estimated'])} & {f(r['lambda_true'],4)} & "
        f"{f(r['lambda_estimated'],4)} & {f(r['anchored_r2'])} \\\\"
        for r in v["anchored_recovery"]["rows"]
    ]
    write(out / "anchored.tex", rows)


def convention(v, out):
    rows = [
        f"{f(r['hurst'],2)} & {f(r['h2_series'])} & {f(r['h2_path'])} & "
        f"{f(r['difference'])} \\\\"
        for r in v["estimand_convention"]["rows"]
    ]
    write(out / "convention.tex", rows)


def finite_size(v, out):
    rows = [
        f"{r['n']} & {f(r['mean_width'])} & {f(r['sd_width'])} & "
        f"{f(r['p95_width'])} \\\\"
        for r in v["finite_size"]["rows"]
    ]
    write(out / "fsize.tex", rows)


def policy(v, out):
    pretty = {
        "fixed_min10seg": "Fixed range, ten-segment floor",
        "adaptive_min10seg": "Adaptive range, ten-segment floor (adopted)",
        "original_L_over_4": r"Adaptive range, $\lfloor L/4\rfloor$ (original)",
    }
    rows = [
        f"{pretty.get(r['policy'], r['policy'])} & "
        f"{f(r['mae_h2_vs_truth'],4)} & {f(r['mean_fabricated_width'],4)} & "
        f"{r['min_segments_at_max_scale']} \\\\"
        for r in v["scaling_policy"]["rows"]
    ]
    write(out / "policy.tex", rows)


def heavy_tail(v, out):
    rows = []
    for r in v["heavy_tail_label"]["rows"]:
        est = ", ".join(f"{x:.3f}" for x in r["h_estimated_q2_to_q5"])
        pw = ", ".join(f"{x:.3f}" for x in r["h_piecewise_q2_to_q5"])
        rows.append(
            f"$H={r['hurst']:.2f}$, $\\gamma={r['gamma']:.1f}$ & "
            f"{est} & {pw} & {f(r['mae_vs_piecewise'],4)} & "
            f"{f(r['mae_vs_flat'],4)} \\\\"
        )
    write(out / "tail.tex", rows)


def standardisation(v, out):
    pretty = {
        "per_order": "Per moment order (adopted)",
        "common": "Common to whole matrix",
        "per_feature": "Per feature",
        "none": "None (not used; see text)",
    }
    rows = [
        f"{pretty.get(r['mode'], r['mode'])} & {f(r['min_column_sd'],3)} & "
        f"{r['n_columns_below_0.01']} & {f(r['ridge_holdout_rmse'],4)} \\\\"
        for r in v["standardisation"]["rows"]
    ]
    write(out / "standardisation.tex", rows)


def benchmark(t, out):
    q = t["q"]
    lstm, mf = t["rmse_by_q"]["lstm"], t["rmse_by_q"]["mfdfa"]
    rows = []
    for target in (-5.0, -2.0, 0.0, 2.0, 5.0):
        i = q.index(target)
        ratio = mf[i] / lstm[i] if lstm[i] > 0 else float("nan")
        rows.append(
            f"${int(target)}$ & {f(lstm[i],4)} & {f(mf[i],4)} & {f(ratio,2)} \\\\"
        )
    rows.append(r"\midrule")
    pl, pm = t["rmse_pooled"]["lstm"], t["rmse_pooled"]["mfdfa"]
    rows.append(
        f"Pooled & {f(pl,4)} & {f(pm,4)} & {f(pm/pl if pl>0 else float('nan'),2)} \\\\"
    )
    rows.append(r"\midrule")
    rows.append(r"\multicolumn{4}{l}{\emph{Pooled by family (LSTM / MF-DFA)}} \\")
    pretty = {
        "fbm": "Fractional Brownian motion",
        "lognormal_mmar": "Lognormal MMAR",
        "binomial": "Binomial cascade",
    }
    for fam, r in t["per_family"].items():
        rows.append(
            f"{pretty.get(fam, fam)} & "
            f"\\multicolumn{{3}}{{l}}{{${f(r['lstm'],4)}$ / ${f(r['mfdfa'],4)}$ "
            f"($n={r['n']}$)}} \\\\"
        )
    write(out / "benchmark.tex", rows)


def anchored_bench(t, out):
    a = t["anchored_rmse"]
    rows = [
        f"$H = h(2)$ & {f(a['H']['lstm'],4)} & {f(a['H']['mfdfa'],4)} \\\\",
        f"$\\lambda$ & {f(a['lambda']['lstm'],4)} & {f(a['lambda']['mfdfa'],4)} \\\\",
    ]
    write(out / "anchoredbench.tex", rows)


def macros(v, t, out):
    """Inline result macros used in running text."""
    b = t["paired_bootstrap_margin"]
    fg = v["fgn_generator"]["rows"]
    lines = [
        "% Generated by scripts/make_report_tables.py -- do not edit by hand.",
        f"\\renewcommand{{\\rLstmPooled}}{{{f(t['rmse_pooled']['lstm'],4)}}}",
        f"\\renewcommand{{\\rMfdfaPooled}}{{{f(t['rmse_pooled']['mfdfa'],4)}}}",
        f"\\renewcommand{{\\rMargin}}{{{f(b['mean'],4)}}}",
        f"\\renewcommand{{\\rMarginLo}}{{{f(b['ci95'][0],4)}}}",
        f"\\renewcommand{{\\rMarginHi}}{{{f(b['ci95'][1],4)}}}",
        f"\\renewcommand{{\\rPathsSeen}}{{{t['corpus']['paths_seen']:,}}}".replace(
            ",", "{,}"
        ),
        f"\\renewcommand{{\\rBestStep}}{{{t['best_step']}}}",
        f"\\renewcommand{{\\rLamLstm}}{{{f(t['anchored_rmse']['lambda']['lstm'],4)}}}",
        f"\\renewcommand{{\\rLamMfdfa}}{{{f(t['anchored_rmse']['lambda']['mfdfa'],4)}}}",
        f"\\renewcommand{{\\rBigHLstm}}{{{f(t['anchored_rmse']['H']['lstm'],4)}}}",
        f"\\renewcommand{{\\rBigHMfdfa}}{{{f(t['anchored_rmse']['H']['mfdfa'],4)}}}",
        # Validation figures quoted in running text.
        f"\\newcommand{{\\rFgnVarTol}}{{{f(max(abs(r['variance']-1) for r in fg),4)}}}",
        f"\\newcommand{{\\rFgnAcfTol}}{{{f(max(r['max_abs_acf_error'] for r in fg),4)}}}",
        f"\\newcommand{{\\rSeqLen}}{{{t['feature_spec']['sequence_length']}}}",
        f"\\newcommand{{\\rNFeatures}}{{{t['feature_spec']['n_features']}}}",
    ]
    (out / "macros.tex").write_text("\n".join(lines) + "\n")
    print(f"  wrote macros.tex ({len(lines)-1} macros)")


PROXY_LABEL = {
    "rv-log-series": r"Log realised volatility, $\log|r_t|$",
    "yz-log-series-w22": r"Log Yang-Zhang, 22-day window",
}


def jse_summary(e, out):
    rows = []
    for key, v in e["primary"].items():
        lab = PROXY_LABEL.get(key, key)
        m, l = v["mfdfa"]["anchored"], v["lstm"]["anchored"]
        rows.append(
            f"{lab} & {f(m['H'])} & {f(m['lambda'],4)} & "
            f"{f(v['mfdfa']['width'])} & {f(l['H'])} & {f(l['lambda'],4)} & "
            f"{f(v['lstm']['width'])} \\\\"
        )
    write(out / "jse_summary.tex", rows)


def jse_diagnostics(e, out):
    rows = []
    for key, v in e["primary"].items():
        lab = PROXY_LABEL.get(key, key)
        s = v["scaling_diagnostics"]
        rows.append(
            f"{lab} & {v['n']} & {f(v['tail']['gamma'],2)} & "
            f"$[{s['scale_min']},\\,{s['scale_max']}]$ & "
            f"{s['segments_at_max_scale']} & {f(s['mean_r2'],4)} \\\\"
        )
    write(out / "jse_diagnostics.tex", rows)


def jse_nulls(e, out):
    pretty = {
        "monofractal": "Monofractal (finite-size)",
        "heavy_tailed": "Rough heavy-tailed",
        "shuffle": "Distributional (shuffle)",
    }
    rows = []
    for key, v in e["primary"].items():
        rows.append(
            f"\\multicolumn{{5}}{{l}}{{\\emph{{{PROXY_LABEL.get(key, key)}}}}} \\\\"
        )
        for kind, t in v["nulls"].items():
            verdict = "reject" if t["p_value"] < 0.05 else "not rejected"
            rows.append(
                f"\\quad {pretty.get(kind, kind)} & {f(t['observed_width'])} & "
                f"{f(t['null_mean'])} & {f(t['null_p95'])} & "
                f"{f(t['p_value'],4)} ({verdict}) \\\\"
            )
    write(out / "jse_nulls.tex", rows)


def jse_surrogates(e, out):
    rows = []
    for key, v in e["primary"].items():
        s = v["surrogates"]
        rows.append(
            f"{PROXY_LABEL.get(key, key)} & {f(s['observed_width'])} & "
            f"{f(s['shuffle_mean'])} & {f(s['iaaft_mean'])} & "
            f"{f(s['delta_pdf'])} & {f(s['delta_nl'])} \\\\"
        )
    write(out / "jse_surrogates.tex", rows)


def jse_convention(c, out):
    pretty = {"rv": "$|r_t|$", "yz": "Yang-Zhang"}
    rows = []
    for key, v in c.items():
        proxy, transform, convention = key.split("-")[0], key.split("-")[1], key.split("-")[2]
        rows.append(
            f"{pretty.get(proxy, proxy)} & "
            f"{'log' if transform == 'log' else 'level'} & {convention} & "
            f"{v['n']} & {f(v['H'])} & {f(v['lambda'],4)} & {f(v['width'])} \\\\"
        )
    write(out / "jse_convention.tex", rows)


def jse_crossindex(e, out):
    rows = []
    for code, v in e["cross_index"].items():
        rows.append(
            f"{code} & {v['n']} & {f(v['gamma'],2)} & "
            f"{f(v['mfdfa']['H'])} & {f(v['mfdfa']['lambda'],4)} & "
            f"{f(v['mfdfa']['width'])} & {f(v['lstm']['H'])} & "
            f"{f(v['lstm']['lambda'],4)} \\\\"
        )
    write(out / "jse_crossindex.tex", rows)


def jse_detrending(e, out):
    rows = [
        f"{order} & {f(v['H'])} & {f(v['lambda'],4)} & {f(v['width'])} & "
        f"$[{v['scale_min']},\\,{v['scale_max']}]$ & {v['segments_at_max_scale']} \\\\"
        for order, v in e["detrending_sensitivity"].items()
    ]
    write(out / "jse_detrending.tex", rows)


def empirical_macros(e, c, out):
    rv = e["primary"]["rv-log-series"]
    lines = [
        "% Generated by scripts/make_report_tables.py -- do not edit by hand.",
        f"\\newcommand{{\\eRvH}}{{{f(rv['mfdfa']['anchored']['H'])}}}",
        f"\\newcommand{{\\eRvLam}}{{{f(rv['mfdfa']['anchored']['lambda'],4)}}}",
        f"\\newcommand{{\\eRvWidth}}{{{f(rv['mfdfa']['width'])}}}",
        f"\\newcommand{{\\eRvGamma}}{{{f(rv['tail']['gamma'],2)}}}",
        f"\\newcommand{{\\eRvDeltaNL}}{{{f(rv['surrogates']['delta_nl'],4)}}}",
        f"\\newcommand{{\\eRvPMono}}{{{f(rv['nulls']['monofractal']['p_value'],4)}}}",
        f"\\newcommand{{\\eRvPTail}}{{{f(rv['nulls']['heavy_tailed']['p_value'],4)}}}",
        f"\\newcommand{{\\eRvPShuf}}{{{f(rv['nulls']['shuffle']['p_value'],4)}}}",
        f"\\newcommand{{\\eRvN}}{{{rv['n']}}}",
        f"\\newcommand{{\\eNNull}}{{{e['n_null']}}}",
        f"\\newcommand{{\\eNSurr}}{{{e['n_surrogates']}}}",
        f"\\newcommand{{\\eMatched}}{{{e['reconciliation']['matched_dates']}}}",
        f"\\newcommand{{\\eRetCorr}}{{{e['reconciliation']['return_correlation']:.5f}}}",
        f"\\newcommand{{\\eFlagged}}{{{e['reconciliation']['n_flagged']}}}",
        f"\\newcommand{{\\eSubstituted}}{{{e['reconciliation']['n_substituted']}}}",
        f"\\newcommand{{\\eYzH}}{{{f(e['primary']['yz-log-series-w22']['mfdfa']['anchored']['H'])}}}",
    ]
    (out / "empirical_macros.tex").write_text("\n".join(lines) + "\n")
    print(f"  wrote empirical_macros.tex ({len(lines)-1} macros)")



def episodes(e, out):
    rows = []
    for p in e["periods"]:
        verdict = "reject" if p["null"]["p_value"] < 0.05 else "not rejected"
        rows.append(
            f"{p['label']} & {p['n']} & {f(p['mfdfa']['H'])} & "
            f"{f(p['mfdfa']['lambda'],4)} & {f(p['mfdfa']['width'])} & "
            f"{f(p['null']['null_mean'])} & {f(p['null']['p_value'],3)} "
            f"({verdict}) \\\\"
        )
    write(out / "episodes.tex", rows)


def episode_macros(e, out):
    d = e["dispersion"]
    lines = [
        "% Generated by scripts/make_report_tables.py -- do not edit by hand.",
        f"\\newcommand{{\\epLamMin}}{{{f(d['lambda_min'],4)}}}",
        f"\\newcommand{{\\epLamMax}}{{{f(d['lambda_max'],4)}}}",
        f"\\newcommand{{\\epLamSd}}{{{f(d['lambda_sd'],4)}}}",
        f"\\newcommand{{\\epWidthRange}}{{{f(d['width_range'])}}}",
        f"\\newcommand{{\\epNullSd}}{{{f(d['mean_null_sd_of_width'])}}}",
        f"\\newcommand{{\\epNNull}}{{{e['n_null']}}}",
    ]
    (out / "episode_macros.tex").write_text("\n".join(lines) + "\n")
    print(f"  wrote episode_macros.tex ({len(lines)-1} macros)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validation", default="results/validation.json")
    ap.add_argument("--train", default="results/benchmark_report.json")
    ap.add_argument("--empirical", default="results/empirical.json")
    ap.add_argument(
        "--convention", default="results/convention_reconciliation.json"
    )
    ap.add_argument("--out", default="report/tables")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    v = json.loads(Path(args.validation).read_text())

    print("validation tables:")
    simulator_labels(v, out)
    anchored(v, out)
    convention(v, out)
    finite_size(v, out)
    policy(v, out)
    heavy_tail(v, out)
    standardisation(v, out)

    train_path = Path(args.train)
    if train_path.exists():
        t = json.loads(train_path.read_text())
        print("benchmark tables:")
        benchmark(t, out)
        anchored_bench(t, out)
        macros(v, t, out)
    else:
        print(f"  {train_path} not found; benchmark tables not generated")

    emp_path, conv_path = Path(args.empirical), Path(args.convention)
    if emp_path.exists():
        e = json.loads(emp_path.read_text())
        print("empirical tables:")
        jse_summary(e, out)
        jse_diagnostics(e, out)
        jse_nulls(e, out)
        jse_surrogates(e, out)
        jse_crossindex(e, out)
        jse_detrending(e, out)
        if conv_path.exists():
            c = json.loads(conv_path.read_text())
            jse_convention(c, out)
            empirical_macros(e, c, out)
    else:
        print(f"  {emp_path} not found; empirical tables not generated")

    ep_path = Path("results/episodes.json")
    if ep_path.exists():
        e2 = json.loads(ep_path.read_text())
        print("episode tables:")
        episodes(e2, out)
        episode_macros(e2, out)


if __name__ == "__main__":
    main()
