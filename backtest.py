"""
backtest.py
=====================================================================
Strictly chronological, day-by-day (no vectorized future peeking)
backtest engine implementing the three required modes, with Indian
delivery-equity transaction costs:

    BUY_COST_RATE  = 0.001187   (0.1187%)
    SELL_COST_RATE = 0.001037   (0.1037%)

Fills are modeled at the SAME-DAY close on which the signal is
generated (consistent with EOD delivery trading conventions used
throughout this dataset's granularity). If your live data feed
provides next-session open, swap `price_col` to that column for a
more realistic fill -- the hook is a plain parameter.

MODE 1 - Stock-level fixed quantity: ₹10,00,000 notional allocated per
         stock, fixed 100-share trades, reports pre- and post-cost P&L.
MODE 2 - Capital-based individual stock: starts at ₹10,00,000, capital
         compounds with trade P&L.
MODE 3 - Portfolio-level: ₹10,00,000 total capital, equally allocated
         across all symbols currently flagged Buy/Strong Buy; closes
         a position whenever that symbol flips to Sell/Strong Sell.

This module is intentionally framework-agnostic (pure pandas / python
loops) so it can be validated independently of NautilusTrader version
quirks. See `nautilus_adapter.py` for the thin shim that runs the same
event loop inside a NautilusTrader Strategy.on_bar() callback.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass

BUY_COST_RATE = 0.001187
SELL_COST_RATE = 0.001037

LONG_SIGNALS = (1, 2)     # Buy, Strong Buy
EXIT_SIGNALS = (-1, -2)   # Sell, Strong Sell


@dataclass
class Trade:
    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    qty: int
    exit_date: pd.Timestamp = None
    exit_price: float = None
    gross_pnl: float = None
    net_pnl: float = None
    buy_cost: float = None
    sell_cost: float = None


# ─────────────────────────────────────────────────────────────────
# MODE 1: stock-level fixed quantity (100 shares), ₹10L allocated capital
# ─────────────────────────────────────────────────────────────────
def backtest_fixed_quantity(symbol: str, sig_df: pd.DataFrame,
                            capital: float = 1_000_000.0, qty: int = 100,
                            price_col: str = "close") -> dict:
    cash = capital
    position = 0
    entry_price = entry_date = None
    cur_buy_cost = 0.0
    trades: list[Trade] = []
    equity_curve = []

    for date, row in sig_df.iterrows():
        price = row[price_col]
        sig = row["signal"]

        if position == 0 and sig in LONG_SIGNALS:
            notional = price * qty
            buy_cost = notional * BUY_COST_RATE
            if cash >= notional + buy_cost:
                cash -= (notional + buy_cost)
                position = qty
                entry_price, entry_date, cur_buy_cost = price, date, buy_cost

        elif position > 0 and sig in EXIT_SIGNALS:
            notional = price * position
            sell_cost = notional * SELL_COST_RATE
            cash += notional - sell_cost
            gross = (price - entry_price) * position
            net = gross - cur_buy_cost - sell_cost
            trades.append(Trade(symbol, entry_date, entry_price, position,
                                date, price, gross, net, cur_buy_cost, sell_cost))
            position = 0

        mtm = cash + (position * price if position else 0.0)
        equity_curve.append((date, mtm))

    if position > 0:
        last_price = sig_df[price_col].iloc[-1]
        notional = last_price * position
        sell_cost = notional * SELL_COST_RATE
        cash += notional - sell_cost
        gross = (last_price - entry_price) * position
        net = gross - cur_buy_cost - sell_cost
        trades.append(Trade(symbol, entry_date, entry_price, position,
                            sig_df.index[-1], last_price, gross, net, cur_buy_cost, sell_cost))

    eq = pd.Series({d: v for d, v in equity_curve})
    gross_pnl = sum(t.gross_pnl for t in trades) if trades else 0.0
    net_pnl = sum(t.net_pnl for t in trades) if trades else 0.0

    return dict(symbol=symbol, mode="fixed_quantity", n_trades=len(trades),
               gross_pnl=gross_pnl, net_pnl=net_pnl,
               final_equity=float(eq.iloc[-1]) if len(eq) else capital,
               total_return_pct=(float(eq.iloc[-1]) / capital - 1) * 100 if len(eq) else 0.0,
               equity_curve=eq, trades=trades)


# ─────────────────────────────────────────────────────────────────
# MODE 2: capital-based individual stock (capital compounds)
# ─────────────────────────────────────────────────────────────────
def backtest_capital_compounding(symbol: str, sig_df: pd.DataFrame,
                                 capital: float = 1_000_000.0,
                                 price_col: str = "close") -> dict:
    cash = capital
    position = 0
    entry_price = entry_date = None
    cur_buy_cost = 0.0
    trades: list[Trade] = []
    equity_curve = []

    for date, row in sig_df.iterrows():
        price = row[price_col]
        sig = row["signal"]

        if position == 0 and sig in LONG_SIGNALS and price > 0:
            max_qty = int(cash // (price * (1 + BUY_COST_RATE)))
            if max_qty > 0:
                notional = price * max_qty
                buy_cost = notional * BUY_COST_RATE
                cash -= (notional + buy_cost)
                position = max_qty
                entry_price, entry_date, cur_buy_cost = price, date, buy_cost

        elif position > 0 and sig in EXIT_SIGNALS:
            notional = price * position
            sell_cost = notional * SELL_COST_RATE
            cash += notional - sell_cost
            gross = (price - entry_price) * position
            net = gross - cur_buy_cost - sell_cost
            trades.append(Trade(symbol, entry_date, entry_price, position,
                                date, price, gross, net, cur_buy_cost, sell_cost))
            position = 0

        mtm = cash + (position * price if position else 0.0)
        equity_curve.append((date, mtm))

    if position > 0:
        last_price = sig_df[price_col].iloc[-1]
        notional = last_price * position
        sell_cost = notional * SELL_COST_RATE
        cash += notional - sell_cost
        gross = (last_price - entry_price) * position
        net = gross - cur_buy_cost - sell_cost
        trades.append(Trade(symbol, entry_date, entry_price, position,
                            sig_df.index[-1], last_price, gross, net, cur_buy_cost, sell_cost))

    eq = pd.Series({d: v for d, v in equity_curve})
    gross_pnl = sum(t.gross_pnl for t in trades) if trades else 0.0
    net_pnl = sum(t.net_pnl for t in trades) if trades else 0.0

    return dict(symbol=symbol, mode="capital_compounding", n_trades=len(trades),
               gross_pnl=gross_pnl, net_pnl=net_pnl,
               final_equity=float(eq.iloc[-1]) if len(eq) else capital,
               total_return_pct=(float(eq.iloc[-1]) / capital - 1) * 100 if len(eq) else 0.0,
               equity_curve=eq, trades=trades)


# ─────────────────────────────────────────────────────────────────
# MODE 3: portfolio-level, equal-weight across all active Buy/Strong-Buy names
# ─────────────────────────────────────────────────────────────────
def backtest_portfolio(sig_by_symbol: dict, capital: float = 1_000_000.0,
                       price_col: str = "close") -> dict:
    all_dates = sorted(set().union(*[df.index for df in sig_by_symbol.values()]))
    cash = capital
    positions: dict = {}
    trades: list[Trade] = []
    equity_curve = []

    for date in all_dates:
        for sym, df in sig_by_symbol.items():
            if sym not in positions or date not in df.index:
                continue
            sig = df.loc[date, "signal"]
            if sig in EXIT_SIGNALS:
                price = df.loc[date, price_col]
                pos = positions.pop(sym)
                notional = price * pos["qty"]
                sell_cost = notional * SELL_COST_RATE
                cash += notional - sell_cost
                gross = (price - pos["entry_price"]) * pos["qty"]
                net = gross - pos["buy_cost"] - sell_cost
                trades.append(Trade(sym, pos["entry_date"], pos["entry_price"], pos["qty"],
                                    date, price, gross, net, pos["buy_cost"], sell_cost))

        candidates = []
        for sym, df in sig_by_symbol.items():
            if sym in positions or date not in df.index:
                continue
            if df.loc[date, "signal"] in LONG_SIGNALS:
                candidates.append(sym)

        if candidates and cash > 0:
            alloc_each = cash / len(candidates)
            for sym in candidates:
                price = sig_by_symbol[sym].loc[date, price_col]
                if price <= 0:
                    continue
                max_qty = int(alloc_each // (price * (1 + BUY_COST_RATE)))
                if max_qty <= 0:
                    continue
                notional = price * max_qty
                buy_cost = notional * BUY_COST_RATE
                cash -= (notional + buy_cost)
                positions[sym] = dict(qty=max_qty, entry_price=price,
                                      entry_date=date, buy_cost=buy_cost)

        mtm = cash
        for sym, pos in positions.items():
            df = sig_by_symbol[sym]
            mtm += pos["qty"] * (df.loc[date, price_col] if date in df.index else pos["entry_price"])
        equity_curve.append((date, mtm))

    for sym, pos in list(positions.items()):
        df = sig_by_symbol[sym]
        last_price = df[price_col].iloc[-1]
        notional = last_price * pos["qty"]
        sell_cost = notional * SELL_COST_RATE
        cash += notional - sell_cost
        gross = (last_price - pos["entry_price"]) * pos["qty"]
        net = gross - pos["buy_cost"] - sell_cost
        trades.append(Trade(sym, pos["entry_date"], pos["entry_price"], pos["qty"],
                            df.index[-1], last_price, gross, net, pos["buy_cost"], sell_cost))

    eq = pd.Series({d: v for d, v in equity_curve})
    gross_pnl = sum(t.gross_pnl for t in trades) if trades else 0.0
    net_pnl = sum(t.net_pnl for t in trades) if trades else 0.0

    return dict(mode="portfolio", n_trades=len(trades),
               gross_pnl=gross_pnl, net_pnl=net_pnl,
               final_equity=float(eq.iloc[-1]) if len(eq) else capital,
               total_return_pct=(float(eq.iloc[-1]) / capital - 1) * 100 if len(eq) else 0.0,
               equity_curve=eq, trades=trades)


def trades_to_frame(trades: list) -> pd.DataFrame:
    rows = [dict(symbol=t.symbol, entry_date=t.entry_date, entry_price=t.entry_price,
                qty=t.qty, exit_date=t.exit_date, exit_price=t.exit_price,
                gross_pnl=t.gross_pnl, net_pnl=t.net_pnl,
                buy_cost=t.buy_cost, sell_cost=t.sell_cost) for t in trades]
    return pd.DataFrame(rows)
