"""
features.py
=====================================================================
Causal (no-look-ahead) feature engineering for the SVR one-day-ahead
price model. Every feature at row t uses ONLY data from rows <= t.

All indicators are hand-rolled (no TA-Lib dependency) so the pipeline
runs anywhere. Output: a DataFrame of features aligned 1:1 with the
input OHLCV DataFrame (same index), plus the regression TARGET
(next-day close), shifted so there is no leakage.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def _macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return macd, sig, hist


def _bollinger(close: pd.Series, window=20, n_std=2.0):
    ma = close.rolling(window).mean()
    sd = close.rolling(window).std()
    upper = ma + n_std * sd
    lower = ma - n_std * sd
    pctb = (close - lower) / (upper - lower).replace(0, np.nan)
    width = (upper - lower) / ma.replace(0, np.nan)
    return ma, upper, lower, pctb.fillna(0.5), width


def _atr(high, low, close, window=14):
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


LAG_DAYS = (1, 2, 3, 5, 10)
MA_WINDOWS = (5, 10, 20, 50, 100)
EMA_WINDOWS = (5, 10, 20, 50)
VOL_WINDOWS = (5, 10, 20, 60)
MOM_WINDOWS = (5, 10, 20)

FEATURE_COLUMNS: list[str] = []  # populated by build_features() on first call


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    df: DataFrame indexed by date with columns open, high, low, close, volume
        (lowercase). Must be sorted ascending by date already.

    Returns a NEW DataFrame (same index) with:
      - all engineered feature columns
      - 'target' = next-day close (close.shift(-1))  -> NaN on the last row
      - 'close'  = today's close (kept for convenience / signal logic)

    STRICTLY CAUSAL: every feature column at row t is computable using only
    close/open/high/low/volume up to and including row t. Only 'target' looks
    forward (by construction, since that's literally what we're predicting),
    and it is the caller's job to never use 'target' as an input feature and
    to drop the last row (where target is NaN) from training.
    """
    out = pd.DataFrame(index=df.index)
    c, o, h, l, v = (df["close"].astype(float), df["open"].astype(float),
                      df["high"].astype(float), df["low"].astype(float),
                      df["volume"].astype(float))

    out["close"] = c
    out["open"] = o
    out["high"] = h
    out["low"] = l
    out["volume"] = v
    out["oc_range"] = (c - o) / o.replace(0, np.nan)
    out["hl_range"] = (h - l) / c.replace(0, np.nan)

    log_ret = np.log(c / c.shift(1))
    out["log_ret_1"] = log_ret
    out["ret_1"] = c.pct_change(1)

    for lag in LAG_DAYS:
        out[f"close_lag_{lag}"] = c.shift(lag)
        out[f"ret_lag_{lag}"] = c.pct_change(lag)

    for w in MA_WINDOWS:
        ma = c.rolling(w).mean()
        out[f"sma_{w}"] = ma
        out[f"close_over_sma_{w}"] = c / ma.replace(0, np.nan) - 1.0

    for w in EMA_WINDOWS:
        ema = c.ewm(span=w, adjust=False).mean()
        out[f"ema_{w}"] = ema
        out[f"close_over_ema_{w}"] = c / ema.replace(0, np.nan) - 1.0

    out["rsi_14"] = _rsi(c, 14)

    macd, macd_sig, macd_hist = _macd(c)
    out["macd"] = macd
    out["macd_signal"] = macd_sig
    out["macd_hist"] = macd_hist

    bb_ma, bb_up, bb_lo, bb_pctb, bb_width = _bollinger(c, 20, 2.0)
    out["bb_pctb"] = bb_pctb
    out["bb_width"] = bb_width

    out["atr_14"] = _atr(h, l, c, 14)
    out["atr_pct_14"] = out["atr_14"] / c.replace(0, np.nan)

    for w in VOL_WINDOWS:
        out[f"vol_{w}"] = log_ret.rolling(w).std() * np.sqrt(252)

    for w in MOM_WINDOWS:
        out[f"momentum_{w}"] = c / c.shift(w) - 1.0

    vol_ma20 = v.rolling(20).mean()
    out["vol_over_vma20"] = v / vol_ma20.replace(0, np.nan)
    out["vol_change_1"] = v.pct_change(1)

    # ---- sanitize: a handful of symbols have a bad-data day (zero/near-zero
    # close or volume from a corporate action, holiday glitch, or feed error).
    # pct_change()/ratio features divide by that day's value and produce +/-inf,
    # which downstream silently corrupts the model (StandardScaler/SVR will
    # hard-crash on inf, NOT just give a bad prediction -- so this must be
    # caught here, not left to the caller). Convert to NaN; the existing
    # dropna(subset=feat_cols) in svr_model.py's training-window slice then
    # simply skips those rows from training, and the daily predict step skips
    # predicting on that date (same as any other warmup/NaN day). ----
    feat_only_cols = [c for c in out.columns]
    out[feat_only_cols] = out[feat_only_cols].replace([np.inf, -np.inf], np.nan)

    # next-day target (the only forward-looking column; never used as a feature)
    out["target"] = c.shift(-1)

    global FEATURE_COLUMNS
    FEATURE_COLUMNS = [col for col in out.columns if col not in ("target",)]

    return out


def feature_columns(feat_df: pd.DataFrame) -> list[str]:
    """Columns usable as model inputs (excludes target and raw OHLCV
    which are kept for reference/signal logic but not fed to the SVR
    in raw-price form, since price-level inputs hurt SVR scale-invariance)."""
    exclude = {"target", "open", "high", "low", "close", "volume"}
    return [c for c in feat_df.columns if c not in exclude]