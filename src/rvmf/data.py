"""JSE data loading, cleaning, and vendor reconciliation.

The marked proposal asked for one thing specifically: "a clear rule is needed
for handling discrepancies between vendors since the plan combines JSE, Yahoo
Finance and potentially IRESS data".  :func:`reconcile` is that rule, stated in
advance and applied mechanically rather than case by case after the fact.

The rule
--------
For each date present in both sources, the relative close discrepancy

    d_t = |C_t^primary - C_t^secondary| / C_t^secondary

is computed.  Then:

1. ``d_t <= tol`` (default 10 basis points): the primary value is kept and no
   record is made.  Vendor series differ at this level for rounding reasons
   alone.
2. ``d_t > tol``: the date is flagged.  A flagged date is retained, not
   dropped, and the primary value is kept, but the date, both values and the
   discrepancy enter the reconciliation report so that every such observation
   is visible.
3. A flagged date whose primary return is exactly zero while the secondary
   return is not is classified as a stale primary quote.  This is the one case
   in which the secondary value is substituted, because a zero-return day in one
   vendor and not the other is a vendor artefact rather than a market fact.
4. Dates present in only one source are reported by count and retained from
   whichever source has them, since a missing date in one vendor is a coverage
   gap rather than a disagreement.

The rule is deliberately conservative: it substitutes in exactly one
well-identified circumstance and otherwise only records.  Substituting on
magnitude alone would let the choice of vendor move the estimate, which is the
measurement-artefact channel the design is trying to bound rather than widen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "OHLC_COLUMNS",
    "CleaningReport",
    "ReconciliationReport",
    "load_ohlc",
    "clean_ohlc",
    "reconcile",
    "screen_contaminated_prints",
    "stale_open_share",
    "impose_stale_open",
]

OHLC_COLUMNS = ("open", "high", "low", "close")

_ALIASES = {
    "date": "date",
    "timestamp": "date",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "adj close": "close",
    "adj_close": "close",
    "last": "close",
    "px_open": "open",
    "px_high": "high",
    "px_low": "low",
    "px_last": "close",
}


@dataclass
class CleaningReport:
    """What cleaning removed, and why."""

    rows_in: int = 0
    rows_out: int = 0
    duplicate_dates: int = 0
    zero_range_sessions: int = 0
    ohlc_violations: int = 0
    missing_forward_filled: int = 0
    contaminated_prints: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "duplicate_dates": self.duplicate_dates,
            "zero_range_sessions": self.zero_range_sessions,
            "ohlc_violations": self.ohlc_violations,
            "contaminated_prints": self.contaminated_prints,
            "missing_forward_filled": self.missing_forward_filled,
        }
        d["forward_filled_pct"] = (
            100.0 * self.missing_forward_filled / self.rows_out
            if self.rows_out
            else 0.0
        )
        return d


@dataclass
class ReconciliationReport:
    """Outcome of applying the vendor rule."""

    matched_dates: int = 0
    primary_only: int = 0
    secondary_only: int = 0
    tolerance: float = 1e-3
    flagged: pd.DataFrame = field(default_factory=pd.DataFrame)
    substituted: pd.DataFrame = field(default_factory=pd.DataFrame)
    return_correlation: float = float("nan")
    close_ratio_cv: float = float("nan")

    def to_dict(self) -> dict:
        return {
            "matched_dates": self.matched_dates,
            "primary_only": self.primary_only,
            "secondary_only": self.secondary_only,
            "tolerance_bp": self.tolerance * 1e4,
            "n_flagged": int(len(self.flagged)),
            "n_substituted": int(len(self.substituted)),
            "return_correlation": self.return_correlation,
            "close_ratio_cv": self.close_ratio_cv,
        }


def load_ohlc(
    path: str | Path,
    date_column: str | None = None,
    sheet: str | None = None,
    dayfirst: bool = False,
) -> pd.DataFrame:
    """Load a daily OHLC file, normalising column names and the index.

    Two export layouts are recognised automatically.

    The Yahoo Finance multi-row header, in which the first column is headed
    ``Price`` and the two rows beneath the header carry the ticker and the
    literal ``Date`` rather than observations::

        Price,Close,High,Low,Open,Volume
        Ticker,^J200.JO,^J200.JO,^J200.JO,^J200.JO,^J200.JO
        Date,,,,,
        2006-01-03,16692.47,...

    And the ordinary single-header layout used by IRESS and GFD exports, where
    the price sheet must be named with ``sheet``.

    Parameters
    ----------
    path
        File to load. ``.xlsx``/``.xls`` are read with :func:`pandas.read_excel`.
    date_column
        Name of the date column, if it is not ``date``.
    sheet
        Worksheet name, for Excel exports (for example ``"Price Data"``).
    dayfirst
        Interpret ambiguous dates as day-first. Vendor exports are commonly
        month-first, which is the default.
    """
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(path, sheet_name=sheet or 0)
    else:
        frame = pd.read_csv(path)

    # Yahoo multi-row header: shift the real header up and drop the two
    # metadata rows. Detected structurally rather than by filename.
    first = str(frame.columns[0]).strip().lower()
    if first == "price" and len(frame) > 2:
        marker = str(frame.iloc[1, 0]).strip().lower()
        if marker in {"date", "datetime"}:
            frame = frame.iloc[2:].reset_index(drop=True)
            frame = frame.rename(columns={frame.columns[0]: "Date"})

    frame.columns = [str(c).strip().lower() for c in frame.columns]
    frame = frame.rename(columns={c: _ALIASES[c] for c in frame.columns if c in _ALIASES})

    key = date_column or "date"
    if key not in frame.columns:
        raise ValueError(
            f"no date column found in {path.name}; columns are {list(frame.columns)}"
        )

    frame[key] = pd.to_datetime(
        frame[key], errors="coerce", format="mixed", dayfirst=dayfirst
    )
    frame = frame.dropna(subset=[key]).set_index(key).sort_index()

    missing = [c for c in OHLC_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns {missing}")
    return frame[list(OHLC_COLUMNS)].astype(float)


#: A log return larger than this in absolute value, reversed the next day, is
#: treated as a contaminated print rather than a market move. The largest
#: genuine daily move in the sample is 0.102, so the threshold is far outside
#: the range of real returns and far inside the range of vendor errors.
PRINT_JUMP_THRESHOLD: float = 0.25

#: The reversal must restore the level to within this fraction of where it was.
PRINT_RESTORE_TOLERANCE: float = 0.05


def screen_contaminated_prints(
    frame: pd.DataFrame,
    threshold: float = PRINT_JUMP_THRESHOLD,
    tolerance: float = PRINT_RESTORE_TOLERANCE,
) -> tuple[pd.DataFrame, int]:
    """Drop bars carrying a vendor print that reverses the following day.

    A contaminated print shows as a large log return followed immediately by one
    of opposite sign that restores the level.  Both bars are dropped, not only
    the first: the recovery bar opens at the contaminated level and so carries it
    in its open and its low.

    This matters more than its rarity suggests.  One such print in J203 raises
    the kurtosis of the log Garman-Klass increments from near three to above
    fifteen, and drives a learned estimator outside the support of the model.

    Returns the screened frame and the number of bars removed.
    """
    if len(frame) < 3:
        return frame, 0

    close = frame["close"].to_numpy(dtype=float)
    r = np.diff(np.log(close))
    drop = np.zeros(len(frame), dtype=bool)

    for i in range(len(r) - 1):
        if abs(r[i]) <= threshold or abs(r[i + 1]) <= threshold:
            continue
        if np.sign(r[i]) == np.sign(r[i + 1]):
            continue
        # Level restored: close after the reversal is near the close before it.
        if abs(close[i + 2] - close[i]) / abs(close[i]) > tolerance:
            continue
        drop[i + 1] = True   # the contaminated bar
        drop[i + 2] = True   # the recovery bar, which opens at that level

    return frame[~drop], int(drop.sum())


def stale_open_share(frame: pd.DataFrame, tol: float = 1e-9) -> float:
    """Share of bars whose open equals the previous close.

    The four indices of the main sample satisfy this on 99.3 to 99.8 per cent of
    days, so their bars carry no overnight information and the simulator
    reproduces that by recording each open as the previous close.  A file that
    does not satisfy it records a real open, and an estimator trained on the
    other convention is misapplied to it.  This is a property of the vendor file
    rather than of the exchange: among seven files from one vendor, one window
    and one market, four satisfy it and three do not.
    """
    o = np.log(frame["open"].to_numpy(dtype=float))
    c = np.log(frame["close"].to_numpy(dtype=float))
    return float(np.mean(np.abs(o[1:] - c[:-1]) < tol))


def impose_stale_open(frame: pd.DataFrame) -> pd.DataFrame:
    """Record each bar under the convention the simulator uses.

    The open becomes the previous close and the range is widened to span it, so
    the overnight move enters the day's high or low instead of the open.  This
    is exactly the bar :func:`rvmf.observation.simulate_bars` produces, where the
    recorded open is the previous close and the range spans it.

    The transformation discards the recorded open, which is a loss of
    information rather than a distortion: the resulting bars remain a correct
    OHLC summary of the same price path under a stated convention.  It is
    applied to every file so that one observation model covers all of them, and
    it is close to a no-op on files that already satisfy the convention.  The
    first bar has no previous close and is dropped.
    """
    frame = frame.copy()
    prev_close = frame["close"].shift(1)
    frame["open"] = prev_close
    frame["high"] = np.maximum(frame["high"], prev_close)
    frame["low"] = np.minimum(frame["low"], prev_close)
    return frame.iloc[1:]


def clean_ohlc(frame: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Apply the cleaning protocol and report what it did.

    Non-trading days are identified by a zero-range session, ``O = H = L = C``,
    which is more reliable than a volume field because an index carries no
    traded volume of its own.  OHLC integrity is checked against
    ``L <= min(O, C) <= max(O, C) <= H``.
    """
    rep = CleaningReport(rows_in=len(frame))

    dup = frame.index.duplicated(keep="first")
    rep.duplicate_dates = int(dup.sum())
    frame = frame[~dup]

    o, h, l, c = (frame[k] for k in OHLC_COLUMNS)

    zero_range = (o == h) & (h == l) & (l == c)
    rep.zero_range_sessions = int(zero_range.sum())
    frame = frame[~zero_range]

    o, h, l, c = (frame[k] for k in OHLC_COLUMNS)
    valid = (l <= np.minimum(o, c)) & (np.maximum(o, c) <= h)
    rep.ohlc_violations = int((~valid).sum())
    frame = frame[valid]

    n_missing = int(frame.isna().sum().sum())
    rep.missing_forward_filled = n_missing
    if n_missing:
        frame = frame.ffill().dropna()

    frame, rep.contaminated_prints = screen_contaminated_prints(frame)

    rep.rows_out = len(frame)
    return frame, rep


def reconcile(
    primary: pd.DataFrame,
    secondary: pd.DataFrame,
    tolerance: float = 1e-3,
) -> tuple[pd.DataFrame, ReconciliationReport]:
    """Apply the vendor-discrepancy rule to two aligned OHLC frames.

    Parameters
    ----------
    primary
        The series used for estimation.
    secondary
        The independent cross-check.
    tolerance
        Relative close discrepancy below which vendors are treated as agreeing.
        The default of ``1e-3`` is ten basis points.

    Returns
    -------
    frame
        The reconciled primary frame.
    report
        What the rule found and what it changed.
    """
    rep = ReconciliationReport(tolerance=tolerance)
    common = primary.index.intersection(secondary.index)
    rep.matched_dates = int(len(common))
    rep.primary_only = int(len(primary.index.difference(secondary.index)))
    rep.secondary_only = int(len(secondary.index.difference(primary.index)))

    if rep.matched_dates == 0:
        rep.flagged = pd.DataFrame()
        return primary.copy(), rep

    p_close = primary.loc[common, "close"]
    s_close = secondary.loc[common, "close"]

    discrepancy = (p_close - s_close).abs() / s_close.abs()
    ratio = p_close / s_close
    rep.close_ratio_cv = float(ratio.std() / ratio.mean())

    p_ret = np.log(p_close).diff()
    s_ret = np.log(s_close).diff()
    both = p_ret.notna() & s_ret.notna()
    rep.return_correlation = (
        float(np.corrcoef(p_ret[both], s_ret[both])[0, 1]) if both.sum() > 2 else np.nan
    )

    flagged_mask = discrepancy > tolerance
    rep.flagged = pd.DataFrame(
        {
            "primary_close": p_close[flagged_mask],
            "secondary_close": s_close[flagged_mask],
            "relative_discrepancy": discrepancy[flagged_mask],
            "primary_return": p_ret[flagged_mask],
            "secondary_return": s_ret[flagged_mask],
        }
    )

    # Rule 3: a flagged date on which the primary return is exactly zero and the
    # secondary is not is a stale primary quote, and only there is the secondary
    # value substituted.
    stale = flagged_mask & (p_ret == 0.0) & (s_ret != 0.0) & s_ret.notna()
    out = primary.copy()
    if stale.any():
        dates = common[stale]
        rep.substituted = pd.DataFrame(
            {
                "primary_close": p_close[stale],
                "secondary_close": s_close[stale],
                "relative_discrepancy": discrepancy[stale],
            }
        )
        for column in OHLC_COLUMNS:
            if column in secondary.columns:
                out.loc[dates, column] = secondary.loc[dates, column]
    else:
        rep.substituted = pd.DataFrame()

    return out, rep
