"""
nautilus_strategy.py  —  NautilusTrader v1.224.0 wiring
=====================================================================
Built against the documented v1.224.0 low-level API (BacktestEngine /
BacktestEngineConfig, custom FeeModel, BarDataWrangler, Equity
instruments, CASH account). I do not have nautilus_trader installed in
this environment to import-check this file line-by-line against your
exact build, so before relying on it: run it on one symbol / one short
date range first (see `if __name__ == "__main__"` at the bottom) and
fix any signature drift against your installed package (`pip show
nautilus_trader`, then `python -c "import nautilus_trader; help(...)"`
on anything that errors). The architecture and control flow are
correct per the documented API; the most likely drift points are
exact constructor kwarg names on Instrument/Equity and BarSpecification,
which Nautilus does occasionally rename between point releases.

ARCHITECTURE (per nautilus_adapter.py's recommendation)
---------------------------------------------------------------------
The SVR model retrains MONTHLY, not per-bar, so we do NOT refit inside
on_bar(). Instead:
  1. run_pipeline.py (already built, already tested) produces
     <out_dir>/<SYMBOL>/signals.csv for the full walk-forward backtest
     window -- this IS the monthly-retrain + daily-prediction process,
     executed once, causally, exactly as specified.
  2. This file's Strategy classes simply READ that pre-computed,
     already-causal signal for "today's" bar and submit orders. This
     is industry-standard: heavy model fitting is decoupled from the
     hot bar-event path. In live deployment, step 1 becomes a monthly
     cron/scheduled job that appends to the same signals store instead
     of being recomputed from a static CSV -- the Strategy code below
     does NOT change between backtest and live (only the data source
     for "today's signal" changes, per Nautilus's research-to-live
     parity design goal).

THREE MODES = THREE STRATEGY CONFIGS, SAME ENGINE/EVENT LOOP
---------------------------------------------------------------------
  Mode 1 (FixedQuantityStrategy)   -> fixed 100-share lots
  Mode 2 (CapitalCompoundingStrategy) -> sizes off live account equity
  Mode 3 (PortfolioStrategy)        -> single multi-instrument strategy,
                                        equal-weights TODAY's cash
                                        across TODAY's Buy/Strong-Buy
                                        candidates (mirrors backtest.py
                                        Mode 3 exactly)

INDIAN DELIVERY COSTS AS A CUSTOM FeeModel
---------------------------------------------------------------------
NautilusTrader's FeeModel.get_commission() signature uses Cython-style
prefixed parameter names (Order_order, Quantity_fill_qty, Price_fill_px,
Instrument_instrument) -- this is the documented v1.224.0 pattern for
subclassing FeeModel in Python (see "Instruments" -> custom fee models
in the Nautilus docs). We use the fill's OrderSide to apply the
asymmetric Indian buy/sell delivery rates.
"""
from __future__ import annotations

import os
import sys
from decimal import Decimal

import pandas as pd

# ---- NautilusTrader imports (v1.224.0) -----------------------------
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.models import FeeModel
from nautilus_trader.config import BacktestEngineConfig, LoggingConfig
from nautilus_trader.model.currencies import USD  # fallback if INR not registered
from nautilus_trader.model.data import Bar, BarSpecification, BarType
from nautilus_trader.model.enums import (AccountType, AggregationSource,
                                         BarAggregation, OmsType, OrderSide,
                                         PriceType)
from nautilus_trader.model.identifiers import InstrumentId, Symbol, TraderId, Venue
from nautilus_trader.model.instruments import Equity
from nautilus_trader.model.objects import Currency, Money, Price, Quantity
from nautilus_trader.persistence.wranglers import BarDataWrangler
from nautilus_trader.trading.strategy import Strategy, StrategyConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BUY_COST_RATE = 0.001187
SELL_COST_RATE = 0.001037

NSE = Venue("NSE")


# ─────────────────────────────────────────────────────────────────
# Currency: register INR if your build doesn't ship it already.
# Currency.register() is idempotent-safe to call once at import time;
# if INR is already a built-in in your installed version, this becomes
# a no-op lookup instead (wrap in try/except for safety across builds).
# ─────────────────────────────────────────────────────────────────
def get_inr() -> Currency:
    try:
        from nautilus_trader.model.currencies import INR  # may exist in your build
        return INR
    except Exception:
        pass
    try:
        inr = Currency(code="INR", precision=2, iso4217=356, name="Indian Rupee",
                       currency_type=1)  # currency_type: 1 = FIAT (check your build's CurrencyType enum)
        Currency.register(inr, overwrite=False)
        return inr
    except Exception:
        # last resort so the file still imports cleanly for inspection;
        # FIX THIS against your build before running for real.
        return USD


INR = get_inr()


# ─────────────────────────────────────────────────────────────────
# Custom FeeModel: asymmetric Indian delivery-equity buy/sell rates
# ─────────────────────────────────────────────────────────────────
class IndianDeliveryFeeModel(FeeModel):
    """0.1187% on buy fills, 0.1037% on sell fills, charged in the
    instrument's quote currency on fill notional (price * qty)."""

    def get_commission(self, Order_order, Quantity_fill_qty, Price_fill_px,
                       Instrument_instrument):
        notional = float(Price_fill_px) * float(Quantity_fill_qty)
        rate = BUY_COST_RATE if Order_order.side == OrderSide.BUY else SELL_COST_RATE
        commission = notional * rate
        return Money(commission, Instrument_instrument.quote_currency)


# ─────────────────────────────────────────────────────────────────
# Instrument + data helpers
# ─────────────────────────────────────────────────────────────────
def make_equity_instrument(symbol: str) -> Equity:
    """Synthetic NSE equity instrument (no live InstrumentProvider for
    NSE equities ships with core Nautilus, so we define it directly --
    this is the documented pattern: "or by the user via the Instrument
    constructor / its subclasses"). Adjust price_precision/increment
    to match NSE tick size (paise -> 0.01 is standard for cash equity)."""
    iid = InstrumentId(symbol=Symbol(symbol), venue=NSE)
    return Equity(
        instrument_id=iid,
        raw_symbol=Symbol(symbol),
        currency=INR,
        price_precision=2,
        price_increment=Price(0.01, precision=2),
        lot_size=Quantity(1, precision=0),
        ts_event=0,
        ts_init=0,
    )


DAILY_BAR_SPEC = BarSpecification(step=1, aggregation=BarAggregation.DAY,
                                  price_type=PriceType.LAST)


def bar_type_for(instrument_id: InstrumentId) -> BarType:
    return BarType(instrument_id=instrument_id, bar_spec=DAILY_BAR_SPEC,
                   aggregation_source=AggregationSource.EXTERNAL)


def load_bars_for_instrument(csv_path: str, instrument: Equity) -> list:
    """csv_path -> the SAME OHLCV CSV used by run_pipeline.py
    (date, open, high, low, close, volume). Returns a list of Bar
    objects via BarDataWrangler, ready for engine.add_data()."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    date_col = next((c for c in df.columns if "date" in c), "date")
    df[date_col] = pd.to_datetime(df[date_col], format="mixed")
    df = df.set_index(date_col).sort_index()
    df = df[["open", "high", "low", "close", "volume"]].dropna()

    bar_type = bar_type_for(instrument.id)
    wrangler = BarDataWrangler(bar_type=bar_type, instrument=instrument)
    bars = wrangler.process(df)
    return bars


def load_signal_lookup(signals_csv: str) -> dict:
    """signals.csv comes straight out of run_pipeline.py / signals.py.
    Returns {pandas.Timestamp.date(): signal_int}."""
    df = pd.read_csv(signals_csv, index_col=0, parse_dates=True)
    return {d.date(): int(s) for d, s in df["signal"].items()}


LONG_SIGNALS = (1, 2)
EXIT_SIGNALS = (-1, -2)


# ─────────────────────────────────────────────────────────────────
# MODE 1: fixed-quantity strategy
# ─────────────────────────────────────────────────────────────────
class FixedQtyConfig(StrategyConfig, frozen=True):
    instrument_id: str
    signals_csv: str
    trade_qty: int = 100


class FixedQuantityStrategy(Strategy):
    def __init__(self, config: FixedQtyConfig):
        super().__init__(config)
        self.signal_by_date = load_signal_lookup(config.signals_csv)
        self.instrument_id = InstrumentId.from_str(config.instrument_id)

    def on_start(self):
        instrument = self.cache.instrument(self.instrument_id)
        if instrument is None:
            self.log.error(f"Instrument {self.instrument_id} not found in cache")
            self.stop()
            return
        self.subscribe_bars(bar_type_for(self.instrument_id))

    def on_bar(self, bar: Bar):
        bar_date = pd.Timestamp(bar.ts_event, unit="ns").date()
        sig = self.signal_by_date.get(bar_date)
        if sig is None:
            return

        instrument = self.cache.instrument(self.instrument_id)
        position = self.cache.position_for_instrument(self.instrument_id) if hasattr(
            self.cache, "position_for_instrument") else None
        net_qty = self.portfolio.net_position(self.instrument_id)

        if net_qty == 0 and sig in LONG_SIGNALS:
            order = self.order_factory.market(
                instrument_id=self.instrument_id, order_side=OrderSide.BUY,
                quantity=Quantity(self.config.trade_qty, precision=0),
            )
            self.submit_order(order)
        elif net_qty > 0 and sig in EXIT_SIGNALS:
            order = self.order_factory.market(
                instrument_id=self.instrument_id, order_side=OrderSide.SELL,
                quantity=Quantity(abs(net_qty), precision=0),
            )
            self.submit_order(order)

    def on_stop(self):
        # Force-close any open position at backtest/session end so results
        # are directly comparable to backtest.py's force-liquidation-at-
        # last-bar behavior, and so summary PnL stats reflect a fully
        # realized result rather than a stale mark-to-market position.
        net_qty = self.portfolio.net_position(self.instrument_id)
        if net_qty > 0:
            order = self.order_factory.market(
                instrument_id=self.instrument_id, order_side=OrderSide.SELL,
                quantity=Quantity(abs(net_qty), precision=0),
            )
            self.submit_order(order)


# ─────────────────────────────────────────────────────────────────
# MODE 2: capital-compounding strategy (sizes off live account equity)
# ─────────────────────────────────────────────────────────────────
class CapitalCompoundingConfig(StrategyConfig, frozen=True):
    instrument_id: str
    signals_csv: str


class CapitalCompoundingStrategy(Strategy):
    def __init__(self, config: CapitalCompoundingConfig):
        super().__init__(config)
        self.signal_by_date = load_signal_lookup(config.signals_csv)
        self.instrument_id = InstrumentId.from_str(config.instrument_id)

    def on_start(self):
        self.subscribe_bars(bar_type_for(self.instrument_id))

    def on_bar(self, bar: Bar):
        bar_date = pd.Timestamp(bar.ts_event, unit="ns").date()
        sig = self.signal_by_date.get(bar_date)
        if sig is None:
            return

        net_qty = self.portfolio.net_position(self.instrument_id)
        price = float(bar.close)

        if net_qty == 0 and sig in LONG_SIGNALS and price > 0:
            account = self.portfolio.account(NSE)
            free_cash = float(account.balance_free(INR).as_double()) if account else 0.0
            max_qty = int(free_cash // (price * (1 + BUY_COST_RATE)))
            if max_qty > 0:
                order = self.order_factory.market(
                    instrument_id=self.instrument_id, order_side=OrderSide.BUY,
                    quantity=Quantity(max_qty, precision=0),
                )
                self.submit_order(order)
        elif net_qty > 0 and sig in EXIT_SIGNALS:
            order = self.order_factory.market(
                instrument_id=self.instrument_id, order_side=OrderSide.SELL,
                quantity=Quantity(abs(net_qty), precision=0),
            )
            self.submit_order(order)

    def on_stop(self):
        # Same rationale as FixedQuantityStrategy.on_stop(): without this,
        # a position still open at backtest end stays unrealized and
        # distorts PnL/Sharpe/win-rate stats, and makes this engine's
        # output non-comparable to backtest.py's forced final-bar exit.
        net_qty = self.portfolio.net_position(self.instrument_id)
        if net_qty > 0:
            order = self.order_factory.market(
                instrument_id=self.instrument_id, order_side=OrderSide.SELL,
                quantity=Quantity(abs(net_qty), precision=0),
            )
            self.submit_order(order)


# ─────────────────────────────────────────────────────────────────
# MODE 3: portfolio-level, equal-weight across all active Buy/Strong-Buy
# names, single multi-instrument strategy sharing one CASH account.
# ─────────────────────────────────────────────────────────────────
class PortfolioConfig(StrategyConfig, frozen=True):
    instrument_ids: tuple   # tuple[str, ...]
    signals_csv_by_symbol: dict  # {symbol: path}


class PortfolioStrategy(Strategy):
    def __init__(self, config: PortfolioConfig):
        super().__init__(config)
        self.instrument_ids = [InstrumentId.from_str(s) for s in config.instrument_ids]
        self.signal_by_symbol = {
            iid.symbol.value: load_signal_lookup(config.signals_csv_by_symbol[iid.symbol.value])
            for iid in self.instrument_ids
        }
        self._n_instruments = len(self.instrument_ids)
        # NOTE: entries are no longer gated on seeing a bar from EVERY
        # instrument on a given date. With instruments that have different
        # listing histories / trading calendars (e.g. INDIGO has ~2,600
        # rows vs RELIANCE's ~6,500), requiring a full synchronized bar
        # set per date silently drops entries on any date where even one
        # symbol has a calendar gap -- across a 7.5y multi-symbol backtest
        # this suppressed a large fraction of trades and diverged sharply
        # from backtest.py's per-date, per-symbol-independent logic (which
        # only checks `date in df.index` per symbol, never requires all
        # symbols to co-report). Instead we accumulate bars PER DATE
        # (keyed by date, never overwritten) and flush a date's entries
        # the first time we observe the date roll forward past it.
        #
        # IMPORTANT: bars across instruments are not guaranteed to arrive
        # in perfectly clean, strictly-increasing-by-calendar-date order
        # even after engine.sort_data() -- tiny ts_event differences
        # between symbols can cause the date sequence to interleave near
        # a boundary (e.g. ...,02-04, 02-01, 02-04,...). Without a
        # processed-dates guard this caused _process_entries() to run
        # TWICE for the same date, each time computing a BUY against a
        # stale "net_qty == 0" + a stale free-cash snapshot (since the
        # first order's fill hadn't landed yet), double-spending the same
        # cash pool and driving the account negative
        # (AccountBalanceNegative engine abort). _processed_dates makes
        # that scenario a strict no-op on the second attempt.
        self._current_date = None
        self._bars_by_date = {}      # date -> {symbol: bar}, accumulates, never cleared
        self._processed_dates = set()

    def on_start(self):
        for iid in self.instrument_ids:
            self.subscribe_bars(bar_type_for(iid))

    def on_bar(self, bar: Bar):
        bar_date = pd.Timestamp(bar.ts_event, unit="ns").date()
        symbol = bar.bar_type.instrument_id.symbol.value
        iid = bar.bar_type.instrument_id

        # Date rollover: flush entries for the previous date using
        # whatever bars were accumulated for it so far. Guarded by
        # _processed_dates so a date already flushed is never re-flushed,
        # even if a later bar arrives whose date briefly steps backward.
        if self._current_date is not None and bar_date != self._current_date:
            self._process_entries(self._current_date)
        self._current_date = bar_date
        self._bars_by_date.setdefault(bar_date, {})[symbol] = bar

        # Exits are independent of the allocation batching and can be
        # processed immediately, same as backtest.py's exit loop (which
        # also runs ahead of that date's entry loop).
        sig = self.signal_by_symbol.get(symbol, {}).get(bar_date)
        net_qty = self.portfolio.net_position(iid)
        if net_qty > 0 and sig in EXIT_SIGNALS:
            order = self.order_factory.market(
                instrument_id=iid, order_side=OrderSide.SELL,
                quantity=Quantity(abs(net_qty), precision=0),
            )
            self.submit_order(order)

    def _process_entries(self, bar_date):
        if bar_date in self._processed_dates:
            return   # hard guard: never double-spend the same date's cash pool
        self._processed_dates.add(bar_date)

        bars_today = self._bars_by_date.get(bar_date, {})
        candidates = []
        for iid in self.instrument_ids:
            symbol = iid.symbol.value
            if symbol not in bars_today:
                continue   # this symbol had no bar on bar_date -- skip it,
                           # same as backtest.py's `date not in df.index`
            sig = self.signal_by_symbol.get(symbol, {}).get(bar_date)
            net_qty = self.portfolio.net_position(iid)
            if net_qty == 0 and sig in LONG_SIGNALS:
                candidates.append(iid)
        if not candidates:
            return

        account = self.portfolio.account(NSE)
        free_cash = float(account.balance_free(INR).as_double()) if account else 0.0
        if free_cash <= 0:
            return
        alloc_each = free_cash / len(candidates)

        for iid in candidates:
            bar = bars_today.get(iid.symbol.value)
            if bar is None:
                continue
            price = float(bar.close)
            if price <= 0:
                continue
            max_qty = int(alloc_each // (price * (1 + BUY_COST_RATE)))
            if max_qty <= 0:
                continue
            order = self.order_factory.market(
                instrument_id=iid, order_side=OrderSide.BUY,
                quantity=Quantity(max_qty, precision=0),
            )
            self.submit_order(order)

    def on_stop(self):
        # Flush any still-pending final date's entries (mirrors the last
        # iteration of backtest.py's `for date in all_dates` loop, which
        # always completes its entry pass even on the final date).
        if self._current_date is not None:
            self._process_entries(self._current_date)

        # Force-close every still-open position across all instruments at
        # session end -- same rationale as Modes 1/2: without this, open
        # positions remain unrealized and distort the portfolio-level PnL
        # stats, and the result isn't comparable to backtest_portfolio()'s
        # forced final-bar exit in backtest.py.
        for iid in self.instrument_ids:
            net_qty = self.portfolio.net_position(iid)
            if net_qty > 0:
                order = self.order_factory.market(
                    instrument_id=iid, order_side=OrderSide.SELL,
                    quantity=Quantity(abs(net_qty), precision=0),
                )
                self.submit_order(order)


# ─────────────────────────────────────────────────────────────────
# Engine builder
# ─────────────────────────────────────────────────────────────────
def build_engine(capital: float = 1_000_000.0) -> BacktestEngine:
    engine_config = BacktestEngineConfig(
        trader_id=TraderId("QBEAST-SVR-001"),
        logging=LoggingConfig(log_level="INFO"),
    )
    engine = BacktestEngine(config=engine_config)
    engine.add_venue(
        venue=NSE,
        oms_type=OmsType.NETTING,
        account_type=AccountType.CASH,
        base_currency=INR,
        starting_balances=[Money(capital, INR)],
        fee_model=IndianDeliveryFeeModel(),
    )
    return engine


def run_mode1(symbol: str, data_dir: str, results_dir: str, capital: float = 1_000_000.0):
    instrument = make_equity_instrument(symbol)
    engine = build_engine(capital)
    engine.add_instrument(instrument)
    bars = load_bars_for_instrument(os.path.join(data_dir, f"{symbol}.csv"), instrument)
    engine.add_data(bars)

    cfg = FixedQtyConfig(
        instrument_id=str(instrument.id),
        signals_csv=os.path.join(results_dir, symbol, "signals.csv"),
        trade_qty=100,
    )
    engine.add_strategy(FixedQuantityStrategy(cfg))
    engine.run()
    return engine


def run_mode2(symbol: str, data_dir: str, results_dir: str, capital: float = 1_000_000.0):
    instrument = make_equity_instrument(symbol)
    engine = build_engine(capital)
    engine.add_instrument(instrument)
    bars = load_bars_for_instrument(os.path.join(data_dir, f"{symbol}.csv"), instrument)
    engine.add_data(bars)

    cfg = CapitalCompoundingConfig(
        instrument_id=str(instrument.id),
        signals_csv=os.path.join(results_dir, symbol, "signals.csv"),
    )
    engine.add_strategy(CapitalCompoundingStrategy(cfg))
    engine.run()
    return engine


def run_mode3(symbols: list, data_dir: str, results_dir: str, capital: float = 1_000_000.0):
    engine = build_engine(capital)
    instrument_ids = []
    signals_paths = {}
    for sym in symbols:
        instrument = make_equity_instrument(sym)
        engine.add_instrument(instrument)
        bars = load_bars_for_instrument(os.path.join(data_dir, f"{sym}.csv"), instrument)
        engine.add_data(bars, sort=False)
        instrument_ids.append(str(instrument.id))
        signals_paths[sym] = os.path.join(results_dir, sym, "signals.csv")

    # bars were added per-symbol with sort=False (cheaper than re-sorting
    # the whole merged stream after every single-symbol add); the combined
    # multi-instrument stream must be sorted chronologically exactly once,
    # before running, or the engine can't replay events in time order.
    engine.sort_data()

    cfg = PortfolioConfig(instrument_ids=tuple(instrument_ids),
                          signals_csv_by_symbol=signals_paths)
    engine.add_strategy(PortfolioStrategy(cfg))
    engine.run()
    return engine


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["1", "2", "3"], required=True)
    ap.add_argument("--symbols", required=True, help="comma-separated; mode 1/2 use only the first")
    ap.add_argument("--data-dir", default="/mnt/project")
    ap.add_argument("--results-dir", default="/mnt/user-data/outputs/svr_results",
                    help="output of run_pipeline.py -- must already exist (signals.csv per symbol)")
    ap.add_argument("--capital", type=float, default=1_000_000.0)
    args = ap.parse_args()

    syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    if args.mode == "1":
        eng = run_mode1(syms[0], args.data_dir, args.results_dir, args.capital)
    elif args.mode == "2":
        eng = run_mode2(syms[0], args.data_dir, args.results_dir, args.capital)
    else:
        eng = run_mode3(syms, args.data_dir, args.results_dir, args.capital)

    print(eng.trader.generate_account_report(NSE))
    print(eng.trader.generate_order_fills_report())