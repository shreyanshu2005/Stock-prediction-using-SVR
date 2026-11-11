"""
run_pipeline.py
=====================================================================
Main entry point. For each symbol CSV:
  1. Load OHLCV, build causal features (features.py)
  2. Walk-forward SVR train (Jan2016-Dec2018 initial) + monthly retrain
     + daily one-day-ahead prediction, backtested Jan2019-Jun2026
     (svr_model.py) -- one of 3 retrain strategies, switchable via CLI
  3. Generate 5-class signals using qbeast slope/accel/regime (signals.py)
  4. Evaluate prediction quality (metrics.py) + save scatter/rolling plots
  5. Run all 3 backtest modes (backtest.py) with Indian transaction costs

Multiprocessing parallelizes step 2+3+5 ACROSS SYMBOLS (each symbol's
walk-forward loop is itself still strictly sequential/causal -- only
independent symbols run in separate processes).

Usage:
    python run_pipeline.py --symbols RELIANCE,HDFCBANK,MARUTI \
        --data-dir /mnt/project --out-dir /mnt/user-data/outputs/svr_results \
        --retrain-strategy rolling --workers 4

    Retrain strategy options: rolling | incremental | rolling_ewma
"""
from __future__ import annotations
import argparse
import os
import sys
import json
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import build_features
from svr_model import SVRConfig, WalkForwardSVR
from signals import build_signals
from metrics import compute_metrics, rolling_error, plot_scatter, plot_equity_curve
from backtest import (backtest_fixed_quantity, backtest_capital_compounding,
                      backtest_portfolio, trades_to_frame)

INITIAL_TRAIN_START = pd.Timestamp("2016-01-01")
INITIAL_TRAIN_END = pd.Timestamp("2018-12-31")
BACKTEST_START = pd.Timestamp("2019-01-01")
BACKTEST_END = pd.Timestamp("2026-06-30")


def load_symbol_csv(data_dir: str, symbol: str) -> pd.DataFrame:
    path = os.path.join(data_dir, f"{symbol}.csv")
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    date_col = next((c for c in df.columns if "date" in c), "date")
    df[date_col] = pd.to_datetime(df[date_col], format="mixed")
    df = df.set_index(date_col).sort_index()
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    df = df[keep].dropna(subset=["close"])
    df = df[~df.index.duplicated(keep="last")]
    return df


def process_symbol(symbol: str, data_dir: str, out_dir: str,
                   retrain_strategy: str, capital: float) -> dict:
    try:
        sym_out = os.path.join(out_dir, symbol)
        os.makedirs(sym_out, exist_ok=True)

        df = load_symbol_csv(data_dir, symbol)
        if df.index.min() > INITIAL_TRAIN_START or df.index.max() < BACKTEST_START:
            return dict(symbol=symbol, status="skipped_insufficient_history")

        feat = build_features(df)

        cfg = SVRConfig(retrain_strategy=retrain_strategy)
        wf = WalkForwardSVR(cfg)
        pred_df = wf.run(feat, INITIAL_TRAIN_END, BACKTEST_START, BACKTEST_END)
        pred_df.to_csv(os.path.join(sym_out, "predictions.csv"))

        # ---- prediction quality metrics ----
        m = compute_metrics(pred_df)
        roll = rolling_error(pred_df, window=20)
        roll.to_csv(os.path.join(sym_out, "rolling_error.csv"))
        if not pred_df.dropna(subset=["final_pred", "actual_next_close"]).empty:
            plot_scatter(pred_df, os.path.join(sym_out, "pred_vs_actual_scatter.png"),
                        title=f"{symbol}: Predicted vs Actual (1-day-ahead)")

        # ---- signals (uses actual close as anchor + SVR's own-day prediction) ----
        close_bt = df["close"].reindex(pred_df.index)
        sig_df = build_signals(close_bt, pred_df["final_pred"])
        sig_df.to_csv(os.path.join(sym_out, "signals.csv"))

        # ---- backtests (modes 1 & 2; mode 3 is run across symbols in main()) ----
        r1 = backtest_fixed_quantity(symbol, sig_df, capital=capital, qty=100)
        r2 = backtest_capital_compounding(symbol, sig_df, capital=capital)

        plot_equity_curve(r1["equity_curve"],
                          os.path.join(sym_out, "equity_mode1_fixed_qty.png"),
                          title=f"{symbol} Mode 1: Fixed Qty Equity")
        plot_equity_curve(r2["equity_curve"],
                          os.path.join(sym_out, "equity_mode2_capital_compounding.png"),
                          title=f"{symbol} Mode 2: Capital Compounding Equity")

        trades_to_frame(r1["trades"]).to_csv(os.path.join(sym_out, "trades_mode1.csv"), index=False)
        trades_to_frame(r2["trades"]).to_csv(os.path.join(sym_out, "trades_mode2.csv"), index=False)

        summary = dict(symbol=symbol, status="ok", metrics=m,
                       mode1=_strip_curve(r1), mode2=_strip_curve(r2))
        with open(os.path.join(sym_out, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2, default=str)

        # return enough to build the portfolio backtest centrally
        return dict(symbol=symbol, status="ok", metrics=m,
                   mode1=_strip_curve(r1), mode2=_strip_curve(r2),
                   sig_df=sig_df)
    except Exception as e:
        return dict(symbol=symbol, status="error", error=str(e),
                   traceback=traceback.format_exc())


def _strip_curve(result: dict) -> dict:
    return {k: v for k, v in result.items() if k not in ("equity_curve", "trades")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", required=True, help="comma-separated symbol list")
    ap.add_argument("--data-dir", default="/mnt/project")
    ap.add_argument("--out-dir", default="/mnt/user-data/outputs/svr_results")
    ap.add_argument("--retrain-strategy", default="rolling",
                    choices=["rolling", "incremental", "rolling_ewma"])
    ap.add_argument("--capital", type=float, default=1_000_000.0)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    args = ap.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    os.makedirs(args.out_dir, exist_ok=True)

    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_symbol, sym, args.data_dir, args.out_dir,
                          args.retrain_strategy, args.capital): sym for sym in symbols}
        for fut in as_completed(futs):
            sym = futs[fut]
            res = fut.result()
            results[sym] = res
            print(f"[{res.get('status')}] {sym}", flush=True)
            if res.get("status") == "error":
                print(res["traceback"], flush=True)

    # ---- MODE 3: portfolio backtest across all symbols that succeeded ----
    sig_by_symbol = {sym: r["sig_df"] for sym, r in results.items()
                     if r.get("status") == "ok" and "sig_df" in r}
    if sig_by_symbol:
        r3 = backtest_portfolio(sig_by_symbol, capital=args.capital)
        plot_equity_curve(r3["equity_curve"],
                          os.path.join(args.out_dir, "equity_mode3_portfolio.png"),
                          title="Mode 3: Portfolio-Level Equity (Equal-Weight)")
        trades_to_frame(r3["trades"]).to_csv(
            os.path.join(args.out_dir, "trades_mode3_portfolio.csv"), index=False)
        with open(os.path.join(args.out_dir, "mode3_summary.json"), "w") as f:
            json.dump(_strip_curve(r3), f, indent=2, default=str)
        print(f"[portfolio] net_pnl={r3['net_pnl']:.2f}  "
              f"total_return={r3['total_return_pct']:.2f}%  trades={r3['n_trades']}")

    overall = {sym: {k: v for k, v in r.items() if k != "sig_df"} for sym, r in results.items()}
    with open(os.path.join(args.out_dir, "overall_summary.json"), "w") as f:
        json.dump(overall, f, indent=2, default=str)
    print(f"\nDone. Results in: {args.out_dir}")


if __name__ == "__main__":
    main()
