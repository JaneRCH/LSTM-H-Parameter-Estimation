"""Download the daily OHLC data for the seven JSE indices used in the study.

The index files are not redistributed in this repository. This script
rebuilds them from Yahoo Finance in the layout the loader expects
(:func:`rvmf.data.load_ohlc` reads the Yahoo multi-row header directly).

Run from the repository root:

    python data/download_data.py

Files are written to ``data/OHLC_historical_data_<CODE>.csv``.

Note
----
Yahoo Finance can revise historical values, and coverage differs by index
(J210 ends on 2025-09-11 on Yahoo). A download made later may therefore
differ slightly from the snapshot behind the reported results, which was
taken in September and October 2026.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yfinance as yf

INDEX_CODES: tuple[str, ...] = ("J200", "J203", "J210", "J213", "J250", "J258", "J263")
START: str = "2006-01-01"
END: str = "2026-07-01"  # exclusive: the last trading day kept is 2026-06-30

log = logging.getLogger("download_data")


def download_index(code: str, out_dir: Path, start: str = START, end: str = END) -> Path:
    """Download one JSE index from Yahoo Finance and write it as CSV.

    Parameters
    ----------
    code
        JSE index code, for example ``"J200"``. The Yahoo ticker is ``^<code>.JO``.
    out_dir
        Folder that receives ``OHLC_historical_data_<code>.csv``.
    start, end
        Date range passed to :func:`yfinance.download`; ``end`` is exclusive.

    Returns
    -------
    Path
        The file written.
    """
    ticker = f"^{code}.JO"
    # auto_adjust=True matches the snapshot layout: Close, High, Low, Open, Volume.
    frame = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if frame is None or frame.empty:
        raise RuntimeError(f"Yahoo Finance returned no data for {ticker}")
    path = out_dir / f"OHLC_historical_data_{code}.csv"
    frame.to_csv(path)
    log.info("%s: %d rows, %s to %s -> %s", ticker, len(frame),
             frame.index[0].date(), frame.index[-1].date(), path)
    return path


def main() -> None:
    """Download every index listed in ``INDEX_CODES`` (or a subset given on the command line)."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--codes", nargs="*", default=list(INDEX_CODES))
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--start", default=START)
    ap.add_argument("--end", default=END)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for code in args.codes:
        download_index(code, out_dir, args.start, args.end)


if __name__ == "__main__":
    main()
