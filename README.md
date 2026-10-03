# LSTM-H-Parameter-Estimation

Generalised Hurst parameter estimation for South African financial markets.

This repository holds the code for an Honours research project (DSC4830,
BCom Honours in Financial Modelling, University of South Africa). The study
estimates the roughness `H` and the intermittency `lambda^2` of the log S-fBM
volatility model from daily open, high, low and close prices of seven
Johannesburg Stock Exchange indices. A two-branch estimator (an LSTM over the
coarse-grained log volatility proxies plus a small MLP over structure-function
summaries) is trained on simulated bars with known parameters. It is then
benchmarked against a moment (GMM) estimator and multifractal detrended
fluctuation analysis (MF-DFA).

Author: Jane Rungaidzo Chipfakacha

## Repository layout

| Path | Contents |
| --- | --- |
| `src/rvmf/` | The library: simulation, observation model, proxies, features, estimators, MF-DFA. |
| `scripts/` | One script per experiment, plus the scripts that write every table and figure. |
| `tests/` | The test suite (`pytest`). |
| `data/` | `download_data.py`, which fetches the index files. The files themselves are not redistributed. |
| `models/` | Trained estimator checkpoints used for the reported results. |
| `results/` | The raw result files behind every reported table and figure, as JSON. |

Scripts write their working files to `results_v2/` and their tables and
figures to `outputs/`. Both are created on first run and are not tracked.

## Installation

Python 3.11.

```bash
git clone https://github.com/JaneRCH/LSTM-H-Parameter-Estimation.git
cd LSTM-H-Parameter-Estimation
pip install -r requirements.txt
```

All commands below run from the repository root with `PYTHONPATH=src`
(on Windows PowerShell: `$env:PYTHONPATH = "src"`).

## Data

```bash
python data/download_data.py
```

This writes `data/OHLC_historical_data_<CODE>.csv` for J200, J203, J210,
J213, J250, J258 and J263 from Yahoo Finance, 2006 (or first availability)
to 30 June 2026. Yahoo can revise past values, so a fresh download may differ
slightly from the snapshot behind the reported results.

## Reproducing the results

```bash
export PYTHONPATH=src

# 1. Simulator validation and test suite (about 2 minutes)
python -m pytest tests/ -q
python scripts/make_validation_outputs.py

# 2. Identification experiment (about 18 minutes)
python scripts/identification_check.py --reps 60

# 3. Link between the model parameters and H(q) (about 25 minutes)
python scripts/apparent_hq.py --reps 10 --length 5100
python scripts/make_apparent_outputs.py

# 4. Training corpus (about 70 minutes)
python scripts/build_corpus.py --n 40000 --split train
python scripts/build_corpus.py --n 4000 --split test --with-proxy

# 5. Estimator and ablations (about 50 minutes each)
python scripts/train_v2.py --steps 3000 --tag full
python scripts/train_v2.py --steps 3000 --tag no_summary --no-summary
python scripts/train_v2.py --steps 3000 --tag no_sequence --no-sequence

# 6. Comparison and empirical estimates
python scripts/benchmark_v2.py
python scripts/empirical_v2.py --replicates 300

# 7. Identification frontier in window length (about 100 minutes)
python scripts/power_study.py --reps 60 --steps 3000
python scripts/power_lambda.py --null-reps 400 --alt-reps 300

# 8. Sub-period analysis (about 12 minutes)
python scripts/subperiods.py --reps 200 --sets 250 \
    --codes J200 J203 J210 J213 J250 J258 J263

# 9. Dependence between the indices (seconds)
python scripts/index_overlap.py

# 10. Tables and figures
python scripts/make_outputs_v2.py --stage all --data-dir data
python scripts/make_power_outputs.py
python scripts/make_subperiod_outputs.py
```

To skip training (step 5), copy the shipped checkpoints into place first:

```bash
mkdir -p results_v2/net && cp models/theta_*.pt results_v2/net/
```

## Reproducibility

All randomness derives from one master seed, 56233043, through named child
streams of `numpy.random.SeedSequence`. Streams are named rather than
numbered, so adding a stream does not renumber the existing ones.
`scripts/make_validation_outputs.py` prints the derived seed of every stream.
Package versions used for the reported results are pinned in
`requirements.txt`.

## Modules

| Module | Role |
| --- | --- |
| `seeds.py` | Derives every random stream from the master seed. |
| `sfbm.py` | The latent log S-fBM: covariance, analytic structure function, circulant simulation. |
| `observation.py` | Latent variance to daily bars, under the recording convention of the data. |
| `rangeproxy.py` | The three daily range-based volatility proxies. |
| `gmm_lnm.py` | Moment estimator of both parameters with a free noise term, and the naive regression. |
| `features_v2.py` | Inputs to the learned estimator. |
| `corpus_v2.py` | Prior sampling and generation of the labelled corpus. |
| `model_v2.py` | The two-branch estimator and its target standardisation. |
| `data.py` | Index loading, cleaning and the vendor outlier screen. |
| `mfdfa.py` | MF-DFA, cross-checked against the `MFDFA` package of Rydin Gorjão et al. (2022). |

## superseded/

The design in the approved research proposal estimated a scaling exponent of
a volatility proxy. A measurement audit and an identification experiment led
to its replacement by the design above. The modules and scripts implementing
the original design, including its MF-DFA runs, are kept in `superseded/` for
reference. Modules in `src/rvmf/` without the `_v2` suffix belong to that
design and remain because the test suite still covers them.
