"""
qbeast_slope_accel.py
=====================================================================
Price-invariant, hard-bounded SLOPE and ACCELERATION for time series,
plus direction / pace classifiers and regime-aware BUY / SELL signals.

Designed for QBEAST AI  (NSE/BSE equities, 1,200+ symbols).

Core properties (as specified):
  * Price-invariant     : computed on log-price (default) -> per-bar
                          fractional change, comparable across symbols.
  * Hard-bounded (-1,+1): tanh(normalized * scale). Never +/-Inf.
                          ML / DL / RL ready.
  * Single-pass accel   : ONE quadratic (parabola) fit per window gives
                          curvature directly (2*c). No two sequential
                          slopes -> no noise amplification, one warmup.
  * Causal (no forward bias): sliding_window_view over the recent PAST
                          only. Output[t] depends solely on price[..t].
  * Robust              : explicit NaN / Inf / constant-window / short-
                          input guards. Deterministic (no random noise).

Dependencies: numpy (required). pandas optional (Series in -> Series out).

Author: built with Claude as technical co-pilot for the QBEAST-AI.N team.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

try:                                  # optional, only for Series in -> Series out
    import pandas as pd
    _HAS_PANDAS = True
except Exception:                     # pragma: no cover
    _HAS_PANDAS = False


# =====================================================================
# Label constants (numeric-first, ML-friendly)
# =====================================================================
UP, FLAT, DOWN = 1, 0, -1
INCREASING, CONSTANT, DECREASING = 1, 0, -1
BUY, HOLD, SELL = 1, 0, -1

SLOPE_DIR_LABEL = {UP: "Up", FLAT: "Flat", DOWN: "Down"}
ACCEL_PACE_LABEL = {INCREASING: "Increasing", CONSTANT: "Constant", DECREASING: "Decreasing"}
SIGNAL_LABEL = {BUY: "Buy", HOLD: "Hold", SELL: "Sell"}

# Regime vocabularies (must match the pspline plotting-engine output)
MARKET_REGIMES = ("Crashing", "Falling", "Sideways", "Rising", "Rally")
VOL_REGIMES = ("HighVol", "MedVol", "LowVol")

# Market-regime directional bias in [-1, +1]: pushes the score for/against longs.
MARKET_REGIME_BIAS = {
    "Crashing": -1.00,
    "Falling":  -0.50,
    "Sideways":  0.00,
    "Rising":   +0.50,
    "Rally":    +1.00,
}
# Volatility regime -> decision threshold on |final score|.
# Higher vol => demand a stronger signal (fewer whipsaws).
VOL_REGIME_THRESHOLD = {
    "HighVol": 0.55,
    "MedVol":  0.35,
    "LowVol":  0.20,
}


# =====================================================================
# Internal helpers
# =====================================================================
def _as_float_1d(x) -> np.ndarray:
    """Coerce to a 1-D float64 copy. Never mutates the caller's array."""
    a = np.asarray(x, dtype=np.float64).ravel()
    return a.copy()


def _clean_nonfinite(a: np.ndarray, nan_policy: str = "ffill") -> np.ndarray:
    """Replace +/-Inf with NaN, then resolve NaN per policy.

    nan_policy:
      'ffill' : forward-fill last valid value; leading NaNs back-filled
                with the first valid value; all-NaN -> zeros.
      'zero'  : NaN -> 0.0
      'keep'  : leave NaN (downstream windows will skip them)
    """
    a = a.copy()
    a[~np.isfinite(a)] = np.nan
    if nan_policy == "zero":
        a[np.isnan(a)] = 0.0
        return a
    if nan_policy == "keep":
        return a
    if nan_policy != "ffill":
        raise ValueError(f"unknown nan_policy: {nan_policy!r}")

    mask = np.isnan(a)
    if not mask.any():
        return a
    if mask.all():
        return np.zeros_like(a)
    # forward fill
    idx = np.where(~mask, np.arange(a.size), 0)
    np.maximum.accumulate(idx, out=idx)
    a = a[idx]
    # back-fill the leading run of NaNs with the first valid value
    first_valid = int(np.argmax(~mask))
    if first_valid > 0:
        a[:first_valid] = a[first_valid]
    return a


def _to_working_price(price, price_mode: str, nan_policy: str):
    """Return (transformed_series y, original_length n)."""
    p = _as_float_1d(price)
    n = p.size
    p = _clean_nonfinite(p, nan_policy=nan_policy)
    if price_mode == "raw":
        y = p
    elif price_mode == "log":
        # guard non-positive prices before log
        p = np.where(p <= 0.0, np.nan, p)
        p = _clean_nonfinite(p, nan_policy="ffill")
        p = np.where(p <= 0.0, 1e-12, p)        # final safety floor
        y = np.log(p)
    elif price_mode == "normalize":
        # divide each window by its own mean later; here keep raw, flag handled in caller
        y = p
    else:
        raise ValueError(f"unknown price_mode: {price_mode!r}")
    return y, n


def _causal_windows(y: np.ndarray, window: int) -> np.ndarray:
    """Windows of shape (len-window+1, window). Row i -> y[i:i+window];
    its 'current time' is index (i+window-1). No look-ahead."""
    return sliding_window_view(y, window)


def _place_with_warmup(values: np.ndarray, total_len: int, window: int,
                       fill) -> np.ndarray:
    """values are aligned to window-end positions (len = total_len-window+1).
    Pad the first (window-1) warmup slots with `fill`."""
    out = np.full(total_len, fill, dtype=np.float64)
    out[window - 1:] = values
    return out


def _causal_rolling_scale(v: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    """Causal robust scale (std-like) of v over a trailing `window`,
    NaN-aware. out[t] uses v[t-window+1 .. t]. Warmup -> NaN."""
    n = v.size
    out = np.full(n, np.nan)
    if n < window:
        return out
    W = _causal_windows(v, window)
    valid = np.isfinite(W)
    cnt = valid.sum(axis=1)
    safe = np.where(valid, W, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = safe.sum(axis=1) / np.maximum(cnt, 1)
        var = np.where(valid, (W - mean[:, None]) ** 2, 0.0).sum(axis=1) / np.maximum(cnt - 1, 1)
        std = np.sqrt(var)
    std[cnt < min_periods] = np.nan
    out[window - 1:] = std
    return out


def _bound(value: np.ndarray, *, bound_mode: str, fixed_scale: float,
           k_sigma: float, vol_window: int, min_periods: int,
           eps: float = 1e-12) -> np.ndarray:
    """Map an unbounded feature into (-1, +1) via tanh.

    bound_mode:
      'fixed'    : tanh(value * fixed_scale). Deterministic, preserves
                   absolute steepness. You must calibrate fixed_scale to
                   your bar timeframe.
      'adaptive' : tanh(value / (k_sigma * rolling_robust_scale)).
                   Self-normalizing -> comparable across symbols /
                   timeframes without a magic constant. A `k_sigma`-sigma
                   move maps to ~tanh(1)=0.76.
    NaNs in `value` (warmup) propagate as NaN and are filled by the caller.
    """
    if bound_mode == "fixed":
        return np.tanh(value * fixed_scale)
    if bound_mode == "adaptive":
        scale = _causal_rolling_scale(value, vol_window, min_periods)
        scale = np.where((~np.isfinite(scale)) | (scale < eps), np.nan, scale)
        return np.tanh(value / (k_sigma * scale))
    raise ValueError(f"unknown bound_mode: {bound_mode!r}")


# =====================================================================
# 1) SLOPE
# =====================================================================
def slope(price,
          window: int = 5,
          *,
          price_mode: str = "log",
          bound_mode: str = "adaptive",
          fixed_scale: float = 100.0,
          k_sigma: float = 2.0,
          vol_window: int = 60,
          min_periods: int | None = None,
          nan_policy: str = "ffill",
          warmup_fill: float = 0.0,
          return_raw: bool = False):
    """Causal, price-invariant, tanh-bounded slope.

    Method: ordinary-least-squares slope of `y` (log-price by default)
    against a centered integer time grid over each trailing `window`.
    For a regular grid the slope has the closed form
        beta = sum(t_c * y) / sum(t_c^2),   t_c = arange(n) - (n-1)/2
    so no matrix inverse and no degeneracy (denominator = n(n^2-1)/12 > 0).

    On log-price, beta is the per-bar fractional change -> price-invariant.

    Returns bounded slope in (-1, +1). If return_raw, returns
    (bounded, raw_unbounded).
    """
    if window < 2:
        raise ValueError("slope window must be >= 2")
    if min_periods is None:
        min_periods = max(5, vol_window // 3)

    y, n = _to_working_price(price, price_mode, nan_policy)
    if n < window:
        out = np.full(n, warmup_fill)
        return (out, out.copy()) if return_raw else out

    W = _causal_windows(y, window)
    if price_mode == "normalize":
        # per-window mean normalization -> fractional units, price-invariant
        wmean = W.mean(axis=1, keepdims=True)
        W = W / np.where(np.abs(wmean) < 1e-12, 1e-12, wmean)

    t = np.arange(window) - (window - 1) / 2.0
    s2 = float((t * t).sum())                     # = n(n^2-1)/12
    beta = (W @ t) / s2                            # per-bar slope, y-units

    raw = _place_with_warmup(beta, n, window, np.nan)
    bounded = _bound(raw, bound_mode=bound_mode, fixed_scale=fixed_scale,
                     k_sigma=k_sigma, vol_window=vol_window,
                     min_periods=min_periods)
    bounded = np.where(np.isfinite(bounded), bounded, warmup_fill)
    raw_filled = np.where(np.isfinite(raw), raw, warmup_fill)

    bounded = _match_type(price, bounded)
    if return_raw:
        return bounded, _match_type(price, raw_filled)
    return bounded


# =====================================================================
# 2) ACCELERATION  (single-pass quadratic fit)
# =====================================================================
def acceleration(price,
                 window: int = 7,
                 *,
                 price_mode: str = "log",
                 bound_mode: str = "adaptive",
                 fixed_scale: float = 400.0,
                 k_sigma: float = 2.0,
                 vol_window: int = 60,
                 min_periods: int | None = None,
                 nan_policy: str = "ffill",
                 warmup_fill: float = 0.0,
                 return_raw: bool = False):
    """Causal, price-invariant, tanh-bounded acceleration (curvature).

    Method: ONE quadratic fit  y ~ a + b*t_c + c*t_c^2  per trailing
    window, centered grid. Second derivative of the parabola is 2*c,
    which is the acceleration. With a centered grid the normal equations
    decouple so c has the closed form
        c = (n * sum(t2*y) - S2 * sum(y)) / (n*S4 - S2^2)
    where t2 = t_c^2, S2 = sum(t2), S4 = sum(t2^2).

    Single pass => no double warmup, no slope-of-slope noise amplification.
    Window should be >= 3 (default 7; >= 5 recommended for stability).
    """
    if window < 3:
        raise ValueError("acceleration window must be >= 3 (quadratic needs 3 points)")
    if min_periods is None:
        min_periods = max(5, vol_window // 3)

    y, n = _to_working_price(price, price_mode, nan_policy)
    if n < window:
        out = np.full(n, warmup_fill)
        return (out, out.copy()) if return_raw else out

    W = _causal_windows(y, window)
    if price_mode == "normalize":
        wmean = W.mean(axis=1, keepdims=True)
        W = W / np.where(np.abs(wmean) < 1e-12, 1e-12, wmean)

    t = np.arange(window) - (window - 1) / 2.0
    t2 = t * t
    S2 = float(t2.sum())
    S4 = float((t2 * t2).sum())
    det = window * S4 - S2 * S2                     # > 0 for window >= 3

    sum_y = W.sum(axis=1)
    sum_t2y = W @ t2
    c = (window * sum_t2y - S2 * sum_y) / det
    accel = 2.0 * c                                 # second derivative

    raw = _place_with_warmup(accel, n, window, np.nan)
    bounded = _bound(raw, bound_mode=bound_mode, fixed_scale=fixed_scale,
                     k_sigma=k_sigma, vol_window=vol_window,
                     min_periods=min_periods)
    bounded = np.where(np.isfinite(bounded), bounded, warmup_fill)
    raw_filled = np.where(np.isfinite(raw), raw, warmup_fill)

    bounded = _match_type(price, bounded)
    if return_raw:
        return bounded, _match_type(price, raw_filled)
    return bounded


# =====================================================================
# 3) SLOPE DIRECTION  (Up / Flat / Down)
# =====================================================================
def slope_direction(slope_bounded, flat_band: float = 0.08, *, as_label: bool = False):
    """Classify bounded slope into UP(+1) / FLAT(0) / DOWN(-1).

    flat_band: dead-zone half-width on the bounded slope. |s| <= band -> FLAT.
    """
    s = _as_float_1d(slope_bounded)
    out = np.zeros(s.size, dtype=np.int8)
    out[s > flat_band] = UP
    out[s < -flat_band] = DOWN
    if as_label:
        return np.array([SLOPE_DIR_LABEL[int(v)] for v in out], dtype=object)
    return out


# =====================================================================
# 4) ACCELERATION PACE  (Increasing / Constant / Decreasing)
# =====================================================================
def acceleration_pace(accel_bounded, flat_band: float = 0.08, *, as_label: bool = False):
    """Classify bounded acceleration into INCREASING(+1)/CONSTANT(0)/DECREASING(-1).

    Positive curvature => the slope itself is rising (path bending upward).
    """
    a = _as_float_1d(accel_bounded)
    out = np.zeros(a.size, dtype=np.int8)
    out[a > flat_band] = INCREASING
    out[a < -flat_band] = DECREASING
    if as_label:
        return np.array([ACCEL_PACE_LABEL[int(v)] for v in out], dtype=object)
    return out


# =====================================================================
# Regime helpers
# =====================================================================
def _broadcast_regime(regime, n: int, vocab, name: str) -> np.ndarray:
    """Accept a scalar label or an array aligned to price; validate vocab."""
    if isinstance(regime, str):
        arr = np.array([regime] * n, dtype=object)
    else:
        arr = np.asarray(regime, dtype=object).ravel()
        if arr.size != n:
            raise ValueError(f"{name} length {arr.size} != price length {n}")
    bad = set(np.unique(arr)) - set(vocab)
    if bad:
        raise ValueError(f"unknown {name} value(s): {sorted(bad)}; expected {vocab}")
    return arr


# =====================================================================
# 5/6) SIGNAL SCORE + BUY / SELL
# =====================================================================
def signal_score(slope_bounded, accel_bounded, market_regime,
                 *,
                 w_slope: float = 0.6,
                 w_accel: float = 0.4,
                 regime_gain: float = 0.5):
    """Continuous trade score in [-1, +1] from momentum + regime bias.

    momentum = w_slope * slope + w_accel * accel        (weights renormalized)
    final    = clip(momentum + regime_gain * market_bias, -1, +1)

    Positive => long-favourable, negative => short/exit-favourable.
    This continuous score is the RL/DL-friendly target; the discrete
    BUY/HOLD/SELL below is a thresholded view of it.
    """
    s = _as_float_1d(slope_bounded)
    a = _as_float_1d(accel_bounded)
    n = s.size
    wsum = w_slope + w_accel
    momentum = (w_slope * s + w_accel * a) / (wsum if wsum != 0 else 1.0)

    mreg = _broadcast_regime(market_regime, n, MARKET_REGIMES, "market_regime")
    bias = np.array([MARKET_REGIME_BIAS[m] for m in mreg])

    final = np.clip(momentum + regime_gain * bias, -1.0, 1.0)
    return final


def trade_signals(slope_bounded, accel_bounded, market_regime, vol_regime,
                  *,
                  w_slope: float = 0.6,
                  w_accel: float = 0.4,
                  regime_gain: float = 0.5,
                  flat_band: float = 0.08,
                  require_direction_agreement: bool = True,
                  block_buy_in: tuple = ("Crashing",),
                  block_sell_in: tuple = ("Rally",),
                  as_label: bool = False):
    """Full discrete signal: BUY(+1) / HOLD(0) / SELL(-1).

    Logic:
      1. score = signal_score(...)                      (momentum + market bias)
      2. threshold = VOL_REGIME_THRESHOLD[vol_regime]   (higher vol -> stricter)
      3. BUY  if score >= +threshold
         SELL if score <= -threshold
      4. Guards:
         - require_direction_agreement: don't BUY while slope_direction==DOWN,
           don't SELL while slope_direction==UP (avoids fighting the local trend
           purely on curvature).
         - block_buy_in / block_sell_in: hard regime vetoes (e.g. never BUY in
           a 'Crashing' tape, never SELL into a 'Rally') regardless of score.

    Returns the discrete signal array. Use generate_signals() if you want
    the score and components too.
    """
    s = _as_float_1d(slope_bounded)
    a = _as_float_1d(accel_bounded)
    n = s.size

    score = signal_score(s, a, market_regime,
                         w_slope=w_slope, w_accel=w_accel, regime_gain=regime_gain)

    vreg = _broadcast_regime(vol_regime, n, VOL_REGIMES, "vol_regime")
    thr = np.array([VOL_REGIME_THRESHOLD[v] for v in vreg])
    mreg = _broadcast_regime(market_regime, n, MARKET_REGIMES, "market_regime")
    sdir = slope_direction(s, flat_band=flat_band)

    sig = np.zeros(n, dtype=np.int8)
    buy = score >= thr
    sell = score <= -thr

    if require_direction_agreement:
        buy &= (sdir != DOWN)
        sell &= (sdir != UP)

    if block_buy_in:
        buy &= ~np.isin(mreg, np.array(block_buy_in, dtype=object))
    if block_sell_in:
        sell &= ~np.isin(mreg, np.array(block_sell_in, dtype=object))

    sig[buy] = BUY
    sig[sell] = SELL          # if both somehow true, sell guard below
    sig[buy & sell] = HOLD    # conflicting -> stand aside

    if as_label:
        return np.array([SIGNAL_LABEL[int(v)] for v in sig], dtype=object)
    return sig


def buy_signal(slope_bounded, accel_bounded, market_regime, vol_regime, **kw) -> np.ndarray:
    """Boolean BUY mask. Thin wrapper over trade_signals()."""
    return trade_signals(slope_bounded, accel_bounded, market_regime, vol_regime, **kw) == BUY


def sell_signal(slope_bounded, accel_bounded, market_regime, vol_regime, **kw) -> np.ndarray:
    """Boolean SELL mask. Thin wrapper over trade_signals()."""
    return trade_signals(slope_bounded, accel_bounded, market_regime, vol_regime, **kw) == SELL


# =====================================================================
# Orchestrator
# =====================================================================
def compute_all(price, market_regime=None, vol_regime=None,
                *,
                slope_window: int = 5,
                accel_window: int = 7,
                price_mode: str = "log",
                bound_mode: str = "adaptive",
                flat_band: float = 0.08,
                **bound_kw):
    """Compute every feature in one call. Returns a dict of aligned arrays
    (or a pandas.DataFrame if price is a pandas.Series and pandas is present).

    Keys: slope, acceleration, slope_direction, accel_pace,
          [signal_score, signal] when both regimes are provided.
    """
    sl = slope(price, slope_window, price_mode=price_mode,
               bound_mode=bound_mode, **bound_kw)
    ac = acceleration(price, accel_window, price_mode=price_mode,
                      bound_mode=bound_mode, **bound_kw)
    sl_arr = _as_float_1d(sl)
    ac_arr = _as_float_1d(ac)
    out = {
        "slope": sl_arr,
        "acceleration": ac_arr,
        "slope_direction": slope_direction(sl_arr, flat_band=flat_band),
        "accel_pace": acceleration_pace(ac_arr, flat_band=flat_band),
    }
    if market_regime is not None and vol_regime is not None:
        out["signal_score"] = signal_score(sl_arr, ac_arr, market_regime)
        out["signal"] = trade_signals(sl_arr, ac_arr, market_regime, vol_regime,
                                       flat_band=flat_band)

    if _HAS_PANDAS and isinstance(price, pd.Series):
        return pd.DataFrame(out, index=price.index)
    return out


# =====================================================================
# Type helper (Series in -> Series out)
# =====================================================================
def _match_type(original, result: np.ndarray):
    if _HAS_PANDAS and isinstance(original, pd.Series):
        return pd.Series(result, index=original.index)
    return result
