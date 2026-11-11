"""
signals.py
=====================================================================
Five-class signal generation (Strong Buy / Buy / Hold / Sell /
Strong Sell) for the Indian market, combining:
  * SVR predicted next-day return (predicted close vs today's close)
  * qbeast_slope_accel bounded slope + acceleration (imported, NOT
    reimplemented)
  * qbeast_regime_detection market regime + vol regime (imported, NOT
    reimplemented)

Design goal: minimize false signals / whipsaws by requiring AGREEMENT
between the SVR-implied direction and the slope/acceleration momentum
features, gated by market regime (never Strong Buy in Crashing, never
Strong Sell in Rally) and de-sensitized in HighVol (stricter thresholds).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qbeast_slope_accel import (slope, acceleration, slope_direction,
                                acceleration_pace, MARKET_REGIME_BIAS,
                                VOL_REGIME_THRESHOLD)
from qbeast_regime_detection import detect_regimes

STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL = 2, 1, 0, -1, -2
SIGNAL_LABELS = {STRONG_BUY: "Strong Buy", BUY: "Buy", HOLD: "Hold",
                 SELL: "Sell", STRONG_SELL: "Strong Sell"}


def build_signals(close: pd.Series,
                  pred_next_close: pd.Series,
                  *,
                  slope_window: int = 5,
                  accel_window: int = 7,
                  flat_band: float = 0.08,
                  reg_win: int = 20,
                  vol_warmup: int = 60,
                  pred_ret_strong: float = 0.012,   # |implied return| for "strong" tier
                  pred_ret_weak: float = 0.004,     # |implied return| for plain tier
                  hist_vol: pd.Series | None = None) -> pd.DataFrame:
    """
    Parameters
    ----------
    close : actual close price series (the SVR's own "today" anchor),
            index aligned to pred_next_close.
    pred_next_close : SVR's calibrated one-day-ahead close prediction,
            same index as close (prediction MADE at that date, for the
            NEXT day's close -> strictly causal, uses no future data).
    hist_vol : annualised historical volatility series (e.g. 60d rolling
            std of log returns * sqrt(252)). If None, computed here from
            `close` causally.

    Returns a DataFrame indexed like `close` with columns:
      pred_ret, slope, accel, slope_dir, accel_pace,
      market_regime, vol_regime, signal (int), signal_label
    """
    close = close.astype(float)
    pred_next_close = pred_next_close.astype(float)

    if hist_vol is None:
        log_ret = np.log(close / close.shift(1))
        hist_vol = log_ret.rolling(60).std() * np.sqrt(252)

    # --- imported, causal slope/acceleration on actual close (momentum context) ---
    sl = slope(close.values, window=slope_window, price_mode="log", bound_mode="adaptive")
    ac = acceleration(close.values, window=accel_window, price_mode="log", bound_mode="adaptive")
    sdir = slope_direction(sl, flat_band=flat_band)
    apace = acceleration_pace(ac, flat_band=flat_band)

    # --- imported regime detection: needs a "locus" series; use close itself
    # (this module does not require the pspline plot-engine locus -- any
    # smooth-ish causal proxy works; close is the simplest causal choice here) ---
    mkt_regime, vol_regime = detect_regimes(
        locus=close.values, hist_vol=hist_vol.values, index=close.index,
        reg_win=reg_win, vol_warmup=vol_warmup, mode="per_symbol_percentile")

    pred_ret = (pred_next_close / close) - 1.0

    n = len(close)
    sig = np.zeros(n, dtype=np.int8)

    for i in range(n):
        pr = pred_ret.iloc[i]
        if np.isnan(pr):
            sig[i] = HOLD
            continue

        regime = mkt_regime[i]
        vol = vol_regime[i]
        # stricter bar in HighVol (fewer whipsaws)
        vol_mult = {"HighVol": 1.5, "MedVol": 1.0, "LowVol": 0.8}.get(vol, 1.0)
        strong_bar = pred_ret_strong * vol_mult
        weak_bar = pred_ret_weak * vol_mult

        sd_i, ap_i = sdir[i], apace[i]
        momentum_up = sd_i >= 0      # Up or Flat-but-not-down
        momentum_down = sd_i <= 0

        bullish_agree = (pr > 0) and momentum_up
        bearish_agree = (pr < 0) and momentum_down

        s = HOLD
        if bullish_agree and pr >= strong_bar and ap_i >= 0 and regime != "Crashing":
            s = STRONG_BUY
        elif bullish_agree and pr >= weak_bar and regime not in ("Crashing",):
            s = BUY
        elif bearish_agree and pr <= -strong_bar and ap_i <= 0 and regime != "Rally":
            s = STRONG_SELL
        elif bearish_agree and pr <= -weak_bar and regime not in ("Rally",):
            s = SELL
        else:
            s = HOLD

        sig[i] = s

    out = pd.DataFrame({
        "close": close, "pred_next_close": pred_next_close, "pred_ret": pred_ret,
        "slope": sl, "accel": ac, "slope_dir": sdir, "accel_pace": apace,
        "market_regime": mkt_regime, "vol_regime": vol_regime,
        "signal": sig,
    }, index=close.index)
    out["signal_label"] = out["signal"].map(SIGNAL_LABELS)
    return out
