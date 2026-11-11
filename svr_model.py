"""
svr_model.py
=====================================================================
Walk-forward Support Vector Regression (SVR) for one-day-ahead close
price prediction, with three switchable monthly-retraining strategies
and an optional stock-specific monthly calibration factor.

STRICT NO-LOOKAHEAD CONTRACT
  * On trading day t, the model used to predict close[t+1] is trained
    ONLY on rows with index < t's "as-of" training cutoff (see below).
  * Retraining happens once per calendar month end, using only data
    available up to that point. The freshly retrained model is then
    used for EVERY trading day in the following month (no re-fit
    mid-month -> matches "monthly retraining" requirement and avoids
    O(N) refits per day, while multiprocessing parallelizes the
    monthly refits across symbols / parameter grid).

RETRAIN STRATEGIES (config.retrain_strategy)
  'rolling'      : fixed rolling window (config.train_years, default 3)
                   ending at the retrain date. Every month the window
                   slides forward by exactly one month.
  'incremental'  : the very first window is the initial 3y window; every
                   subsequent retrain ADDS only the newly available
                   month of data to the existing training set (an
                   ever-expanding window seeded from the initial 3y).
  'rolling_ewma' : same rolling 3y window as 'rolling', but each sample
                   is weighted by EWMA recency weight with lambda=0.94,
                   i.e. weight_i = (1-lambda) * lambda^(age_in_days),
                   passed to SVR via sample_weight (SVR with RBF kernel
                   in scikit-learn supports sample_weight).

CALIBRATION
  After each retrain, if requested, a simple linear calibration
  (predicted -> actual) is fit on the most recent calibration_window
  trading days of OUT-OF-SAMPLE one-day-ahead predictions (i.e. the
  predictions actually produced walk-forward in the prior month), and
  used to correct future raw SVR predictions:
      calibrated = a * raw_pred + b
  This is recomputed every month, strictly causally (uses only already-
  realized predictions vs. already-realized actuals).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

RetrainStrategy = Literal["rolling", "incremental", "rolling_ewma"]


@dataclass
class SVRConfig:
    train_years: int = 3
    retrain_strategy: RetrainStrategy = "rolling"
    ewma_lambda: float = 0.94
    svr_kernel: str = "rbf"
    svr_C: float = 10.0
    svr_epsilon: float = 0.01
    svr_gamma: str | float = "scale"
    use_calibration: bool = True
    calibration_min_points: int = 15
    min_train_rows: int = 250          # don't even attempt a fit with less
    max_train_rows: int = 3000         # cap for speed; rolling window also caps this


@dataclass
class _MonthlyModel:
    """One trained model + its preprocessing, valid for one calendar month."""
    scaler_x: StandardScaler
    scaler_y: StandardScaler
    model: SVR
    feature_cols: list
    calib_a: float = 1.0
    calib_b: float = 0.0
    train_end_date: pd.Timestamp = None


def _ewma_weights(dates: pd.DatetimeIndex, as_of: pd.Timestamp, lam: float) -> np.ndarray:
    age_days = (as_of - dates).days.values.astype(float)
    age_days = np.clip(age_days, 0, None)
    w = (1.0 - lam) * (lam ** age_days)
    w = w / (w.sum() + 1e-12) * len(w)   # renormalize to mean 1 (keeps C scale sane)
    return w


def _fit_one_model(X_train: np.ndarray, y_train: np.ndarray, sample_weight,
                    cfg: SVRConfig) -> tuple:
    # Safety net: features.py already sanitizes inf->NaN, but guard here too
    # so a future feature/column can never silently crash the SVR fit.
    finite_rows = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train)
    if not finite_rows.all():
        X_train = X_train[finite_rows]
        y_train = y_train[finite_rows]
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)[finite_rows]

    sx = StandardScaler()
    sy = StandardScaler()
    Xs = sx.fit_transform(X_train)
    ys = sy.fit_transform(y_train.reshape(-1, 1)).ravel()
    svr = SVR(kernel=cfg.svr_kernel, C=cfg.svr_C, epsilon=cfg.svr_epsilon,
              gamma=cfg.svr_gamma)
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    svr.fit(Xs, ys, **fit_kwargs)
    return sx, sy, svr


def _slice_training_window(feat: pd.DataFrame, as_of: pd.Timestamp,
                           cfg: SVRConfig, prev_train_start: Optional[pd.Timestamp]
                           ) -> tuple[pd.DataFrame, Optional[pd.Timestamp]]:
    """Return (training rows with index <= as_of and target known, new_train_start).
    Only uses rows strictly BEFORE as_of's own undetermined target, i.e. rows
    where 'target' (next-day close) is already realized as of `as_of`."""
    usable = feat.loc[feat.index <= as_of].copy()
    usable = usable.dropna(subset=["target"])   # last row of full hist has NaN target

    if cfg.retrain_strategy in ("rolling", "rolling_ewma"):
        start = as_of - pd.DateOffset(years=cfg.train_years)
        window = usable.loc[usable.index > start]
        new_start = window.index[0] if len(window) else None
    else:  # incremental
        if prev_train_start is None:
            start = as_of - pd.DateOffset(years=cfg.train_years)
            window = usable.loc[usable.index > start]
            new_start = window.index[0] if len(window) else None
        else:
            window = usable.loc[usable.index >= prev_train_start]
            new_start = prev_train_start

    if len(window) > cfg.max_train_rows:
        window = window.iloc[-cfg.max_train_rows:]

    return window, new_start


class WalkForwardSVR:
    """
    Orchestrates monthly retraining + daily one-step-ahead prediction.

    Usage:
        wf = WalkForwardSVR(cfg)
        results = wf.run(feat_df, train_start, backtest_start, backtest_end)

    `feat_df` must come from features.build_features() (has 'target' col,
    and a 'close' col). Index = trading dates, ascending, no gaps assumed
    beyond what's naturally in the source data (NaNs in indicator warm-up
    are fine; rows are simply skipped for training until indicators exist).
    """

    def __init__(self, cfg: SVRConfig):
        self.cfg = cfg

    def run(self, feat_df: pd.DataFrame, initial_train_end: pd.Timestamp,
            backtest_start: pd.Timestamp, backtest_end: pd.Timestamp) -> pd.DataFrame:
        cfg = self.cfg
        feat_cols = [c for c in feat_df.columns
                     if c not in ("target", "open", "high", "low", "close", "volume")]

        feat_df = feat_df.sort_index()
        all_dates = feat_df.index
        bt_dates = all_dates[(all_dates >= backtest_start) & (all_dates <= backtest_end)]

        current_model: Optional[_MonthlyModel] = None
        current_month_key = None
        prev_train_start = None
        recent_oos_pred = []   # rolling list of (date, raw_pred, actual) for calibration

        records = []

        for date in bt_dates:
            month_key = (date.year, date.month)

            # Retrain at the first trading day we see in a new month, using
            # all data strictly available BEFORE `date` (causal: training
            # cutoff is the previous trading day, never `date` itself or later).
            if month_key != current_month_key:
                loc = feat_df.index.get_loc(date)
                if loc == 0:
                    continue
                as_of = feat_df.index[loc - 1]   # last fully-realized row before today

                train_window, prev_train_start = _slice_training_window(
                    feat_df, as_of, cfg, prev_train_start)
                train_window = train_window.dropna(subset=feat_cols)

                if len(train_window) >= cfg.min_train_rows:
                    X = train_window[feat_cols].values
                    y = train_window["target"].values

                    sample_w = None
                    if cfg.retrain_strategy == "rolling_ewma":
                        sample_w = _ewma_weights(train_window.index, as_of, cfg.ewma_lambda)

                    sx, sy, svr = _fit_one_model(X, y, sample_w, cfg)
                    current_model = _MonthlyModel(sx, sy, svr, feat_cols,
                                                  train_end_date=as_of)

                    # ---- monthly calibration using last month's OOS predictions ----
                    if cfg.use_calibration and len(recent_oos_pred) >= cfg.calibration_min_points:
                        arr = np.array([[p, a] for _, p, a in recent_oos_pred[-60:]])
                        raw_p, act = arr[:, 0], arr[:, 1]
                        if np.std(raw_p) > 1e-9:
                            a_coef, b_coef = np.polyfit(raw_p, act, 1)
                            current_model.calib_a, current_model.calib_b = float(a_coef), float(b_coef)
                current_month_key = month_key

            row = feat_df.loc[[date]]
            if current_model is None or row[feat_cols].isna().any(axis=1).iloc[0]:
                records.append(dict(date=date, raw_pred=np.nan, calibrated_pred=np.nan,
                                    actual_next_close=row["target"].iloc[0] if "target" in row else np.nan,
                                    close_today=row["close"].iloc[0]))
                continue

            Xt = row[feat_cols].values
            Xs = current_model.scaler_x.transform(Xt)
            y_scaled_pred = current_model.model.predict(Xs)
            raw_pred = current_model.scaler_y.inverse_transform(
                y_scaled_pred.reshape(-1, 1)).ravel()[0]
            calibrated = current_model.calib_a * raw_pred + current_model.calib_b

            actual_next = row["target"].iloc[0]
            records.append(dict(date=date, raw_pred=raw_pred, calibrated_pred=calibrated,
                                actual_next_close=actual_next, close_today=row["close"].iloc[0]))

            # Once the actual is realized (i.e. when we get to the NEXT day, the
            # target for `date` becomes known) it's appended for calibration use.
            if not np.isnan(actual_next):
                recent_oos_pred.append((date, raw_pred, actual_next))
                if len(recent_oos_pred) > 80:
                    recent_oos_pred = recent_oos_pred[-80:]

        out = pd.DataFrame.from_records(records).set_index("date")
        pred_col = "calibrated_pred" if cfg.use_calibration else "raw_pred"
        out["final_pred"] = out[pred_col].where(out[pred_col].notna(), out["raw_pred"])
        return out