"""
qbeast_regime_detection.py
=====================================================================
Standalone REGIME DETECTION for QBEAST AI.

Detects BOTH market regime and volatility regime from:
  - actual/pspline locus series (from plot engine)
  - historical volatility series

Output regime names EXACTLY match qbeast_slope_accel.py vocabulary:
  Market : "Crashing", "Falling", "Sideways", "Rising", "Rally"
  Vol    : "HighVol",  "MedVol",  "LowVol"

Causal (no forward bias) — EOD daily trading safe.
Every team member can import and use detect_regimes() directly.

Usage:
    from qbeast_regime_detection import detect_regimes

    mkt, vol = detect_regimes(
        index   = df.index,
        locus   = pspline_locus_array,   # from plot engine
        hist_vol= hist_vol_series,        # annualised HV
    )

    # Then pass directly to qbeast_slope_accel:
    from qbeast_slope_accel import compute_all
    results = compute_all(close, market_regime=mkt, vol_regime=vol)

Author: QBEAST AI team
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# =====================================================================
# Regime name constants — MUST match qbeast_slope_accel.py exactly
# =====================================================================
MARKET_REGIMES = ("Crashing", "Falling", "Sideways", "Rising", "Rally")
VOL_REGIMES    = ("HighVol", "MedVol", "LowVol")

# Default thresholds (% per day, annualised-vol-based)
# These match the main cycler's classify_market_series defaults.
_DEFAULT_MKT_THRESHOLDS = {
    "crash":   -0.20,   # below this -> Crashing
    "falling": -0.05,   # below this -> Falling
    "rising":  +0.05,   # above this -> Rising
    "rally":   +0.20,   # above this -> Rally
    # between falling and rising -> Sideways
}


# =====================================================================
# Internal helpers
# =====================================================================
def _classify_market_value(pct_per_day: float,
                           thresholds: dict) -> str:
    """Map a single %/day value to a market regime label."""
    if np.isnan(pct_per_day):
        return "Sideways"          # warmup default
    c = thresholds["crash"]
    f = thresholds["falling"]
    r = thresholds["rising"]
    ra = thresholds["rally"]
    if   pct_per_day <= c:  return "Crashing"
    elif pct_per_day <= f:  return "Falling"
    elif pct_per_day >= ra: return "Rally"
    elif pct_per_day >= r:  return "Rising"
    else:                   return "Sideways"


def _classify_market_series_percentile(pct_per_day: np.ndarray,
                                       pct_crash: float = 10,
                                       pct_fall:  float = 30,
                                       pct_rise:  float = 70,
                                       pct_rally: float = 90) -> np.ndarray:
    """
    Per-symbol percentile mode — thresholds derived from the symbol's
    own distribution (no cross-symbol bias). Matches main cycler's
    'per_symbol_percentile' REGIME_MODE.
    """
    valid = pct_per_day[np.isfinite(pct_per_day)]
    if len(valid) < 10:
        return np.array(["Sideways"] * len(pct_per_day), dtype=object)

    c  = np.percentile(valid, pct_crash)
    f  = np.percentile(valid, pct_fall)
    r  = np.percentile(valid, pct_rise)
    ra = np.percentile(valid, pct_rally)

    out = np.empty(len(pct_per_day), dtype=object)
    for i, v in enumerate(pct_per_day):
        if   np.isnan(v):  out[i] = "Sideways"
        elif v <= c:        out[i] = "Crashing"
        elif v <= f:        out[i] = "Falling"
        elif v >= ra:       out[i] = "Rally"
        elif v >= r:        out[i] = "Rising"
        else:               out[i] = "Sideways"
    return out


def _classify_vol_value(hv: float, p33: float, p66: float) -> str:
    """Map a single HV value to a vol regime label using expanding terciles."""
    if np.isnan(hv):
        return "MedVol"
    if   hv >= p66: return "HighVol"
    elif hv <= p33: return "LowVol"
    else:           return "MedVol"


# =====================================================================
# Main public function
# =====================================================================
def detect_regimes(
    locus,
    hist_vol,
    index=None,
    *,
    reg_win: int      = 20,
    vol_warmup: int   = 60,
    mode: str         = "per_symbol_percentile",
    thresholds: dict  = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect market and volatility regimes. Fully causal — no forward bias.

    Parameters
    ----------
    locus     : array-like
        P-Spline locus series from the QBEAST plot engine (actual prices,
        not log). Same length as hist_vol.
    hist_vol  : array-like
        Annualised historical volatility (e.g. 252-day rolling std of
        log-returns). Same length as locus.
    index     : array-like, optional
        DatetimeIndex aligned to locus/hist_vol. Used only for pandas
        alignment; not required.
    reg_win   : int, default 20
        Trailing window (bars) for cumulative locus log-return used to
        assess market direction. 20 trading days = ~1 calendar month.
    vol_warmup: int, default 60
        Minimum number of valid HV observations before assigning a vol
        regime (avoids regime noise in the first few months).
    mode      : str
        "per_symbol_percentile" (default) — thresholds from symbol's own
            distribution. Comparable across low/high-beta stocks.
        "absolute" — fixed %/day thresholds from `thresholds` dict.
    thresholds: dict, optional
        Only used when mode="absolute". Keys: crash, falling, rising, rally.

    Returns
    -------
    market_regime : np.ndarray[object]  shape (N,)
        One of: "Crashing", "Falling", "Sideways", "Rising", "Rally"
    vol_regime    : np.ndarray[object]  shape (N,)
        One of: "HighVol", "MedVol", "LowVol"

    Notes
    -----
    * Output names EXACTLY match MARKET_REGIMES / VOL_REGIMES constants
      in qbeast_slope_accel.py — no name mismatch possible.
    * Regime at bar t uses ONLY data up to and including bar t (causal).
    * "Unknown" is never returned — warmup bars get a neutral default
      ("Sideways" / "MedVol") so downstream signal code never crashes.
    """
    # --- prepare locus ---
    loc_arr = np.asarray(locus, dtype=float).ravel()
    N = len(loc_arr)

    if index is not None:
        idx = pd.Index(index)
    else:
        idx = pd.RangeIndex(N)

    loc_s = pd.Series(loc_arr, index=idx)

    # --- market regime ---
    lret    = np.log(loc_s / loc_s.shift(1))
    cum     = lret.rolling(reg_win).sum().values          # causal trailing
    pct_per_day = (cum / float(reg_win)) * 100.0          # %/day scale

    if mode == "per_symbol_percentile":
        mkt = _classify_market_series_percentile(pct_per_day)
    else:
        thr = thresholds if thresholds is not None else _DEFAULT_MKT_THRESHOLDS
        mkt = np.array([_classify_market_value(v, thr) for v in pct_per_day],
                       dtype=object)

    # warmup bars (cum is NaN) -> neutral "Sideways"
    warmup_mask = ~np.isfinite(cum)
    mkt[warmup_mask] = "Sideways"

    # --- vol regime (expanding terciles — causal) ---
    hv  = np.asarray(hist_vol, dtype=float).ravel()
    vol = np.empty(N, dtype=object)

    for i in range(N):
        past = hv[:i + 1]
        past = past[np.isfinite(past)]
        if np.isnan(hv[i]) or len(past) < vol_warmup:
            vol[i] = "MedVol"          # neutral default during warmup
        else:
            p33 = float(np.percentile(past, 33))
            p66 = float(np.percentile(past, 66))
            vol[i] = _classify_vol_value(hv[i], p33, p66)

    return mkt, vol


# =====================================================================
# Convenience: detect + compute all signals in one call
# =====================================================================
def detect_and_signal(
    close,
    locus,
    hist_vol,
    index=None,
    *,
    slope_window: int = 5,
    accel_window: int = 7,
    reg_win: int      = 20,
    vol_warmup: int   = 60,
    mode: str         = "per_symbol_percentile",
):
    """
    One-shot: regime detection + slope/accel/signal computation.

    Requires qbeast_slope_accel.py in the same directory.

    Returns a pandas DataFrame (if pandas available) with columns:
        slope, acceleration, slope_direction, accel_pace,
        signal_score, signal, market_regime, vol_regime
    """
    try:
        from qbeast_slope_accel import compute_all
    except ImportError:
        raise ImportError(
            "qbeast_slope_accel.py not found. "
            "Place it in the same directory as this file."
        )

    mkt, vol = detect_regimes(
        locus, hist_vol, index=index,
        reg_win=reg_win, vol_warmup=vol_warmup, mode=mode
    )

    results = compute_all(
        close,
        market_regime=mkt,
        vol_regime=vol,
        slope_window=slope_window,
        accel_window=accel_window,
    )

    # attach regime arrays
    if isinstance(results, pd.DataFrame):
        results["market_regime"] = mkt
        results["vol_regime"]    = vol
    elif isinstance(results, dict):
        results["market_regime"] = mkt
        results["vol_regime"]    = vol

    return results


# =====================================================================
# Quick self-test
# =====================================================================
if __name__ == "__main__":
    np.random.seed(42)
    n = 500
    price  = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.01))
    locus  = price * (1 + np.random.randn(n) * 0.002)   # mock pspline
    hv     = np.full(n, 0.20) + np.random.randn(n) * 0.02

    mkt, vol = detect_regimes(locus, hv)

    print("Market regime counts:")
    for r in MARKET_REGIMES:
        print(f"  {r:10s}: {(mkt == r).sum()}")
    print("\nVol regime counts:")
    for r in VOL_REGIMES:
        print(f"  {r:10s}: {(vol == r).sum()}")
    print("\nSample (last 5):")
    for i in range(-5, 0):
        print(f"  bar {n+i:4d}  mkt={mkt[i]:10s}  vol={vol[i]}")
