"""
nautilus_adapter.py
=====================================================================
WHY THIS FILE IS A SHIM, NOT A FULL INTEGRATION
---------------------------------------------------------------------
NautilusTrader's Strategy/Actor/ExecutionEngine/Portfolio API surface
changes meaningfully between versions (instrument definitions, bar
types, order factory signatures, venue/account setup). Generating a
large block of NautilusTrader-specific code here without being able
to import-check it against YOUR installed version would likely hand
you code that *looks* plausible but fails on import -- worse than no
code at all for a "production-ready" requirement.

What you DO have, fully working and tested above (backtest.py,
svr_model.py, signals.py, features.py, run_pipeline.py), is a
strictly event-driven, day-by-day, no-look-ahead engine. That is the
hard, correctness-critical part. Wiring it into NautilusTrader is a
mechanical "translate this loop into callbacks" step once your
NautilusTrader version is pinned. This file shows exactly that
translation pattern so you (or I, once you tell me `pip show
nautilus_trader` version) can drop it into a real Strategy class.

THE TRANSLATION PATTERN
---------------------------------------------------------------------
This engine's "event" is one trading day's bar close for one symbol.
In NautilusTrader terms that is exactly Strategy.on_bar(bar) for a
1-day Bar subscription. The mapping is:

    our day-by-day loop step                 NautilusTrader equivalent
    ---------------------------------------  -----------------------------------
    feat_df.loc[date]                        bar.close / your custom indicator
                                              values computed in on_bar()
    monthly retrain check (date.month         on_bar(): check self.clock.utc_now()
      changed)                                  .month != self._last_retrain_month
    WalkForwardSVR prediction for `date`      self._predict(bar) using the SAME
                                                 svr_model.WalkForwardSVR internals
                                                 (call its per-day step directly,
                                                 OR precompute predictions.csv via
                                                 run_pipeline.py and have the
                                                 Strategy simply READ the
                                                 already-computed signal/prediction
                                                 for `bar.ts_event`'s date -- this
                                                 is the recommended pattern for a
                                                 model that retrains MONTHLY, since
                                                 there is no benefit to refitting
                                                 inside the hot bar-event path)
    build_signals() -> signal int             self.signal = signals.build_signals(...)
                                                 (incrementally extend the causal
                                                 slope/accel/regime series bar-by-bar,
                                                 or again, precompute + look up)
    backtest_fixed_quantity() entry/exit      self.submit_order(MarketOrder(...))
      logic (LONG_SIGNALS / EXIT_SIGNALS)        gated by the same signal value,
                                                 with qty fixed at 100 (Mode 1) or
                                                 sized off self.portfolio.cash()
                                                 (Mode 2), and order side BUY/SELL
    BUY_COST_RATE / SELL_COST_RATE            registered as a FeeModel /
                                                 commission model on the venue
                                                 in your BacktestEngineConfig --
                                                 NautilusTrader has first-class
                                                 support for percentage commission
                                                 models (see your installed
                                                 version's `nautilus_trader.model
                                                 .fees` or `accounting` modules).
    equity_curve list                         self.portfolio.account(venue)
                                                 .balance_total() sampled via
                                                 NautilusTrader's built-in
                                                 PortfolioAnalyzer / report, or a
                                                 custom Actor that logs balance on
                                                 every bar.

RECOMMENDED PRODUCTION ARCHITECTURE
---------------------------------------------------------------------
Because the SVR model retrains MONTHLY (not per-tick/per-bar), the
fastest path to "production-ready, minimal code changes from backtest
to live" is:

  1. A separate, scheduled job (cron / NautilusTrader's own
     `clock.set_time_alert` on a monthly cadence) runs
     run_pipeline.py's per-symbol training step and writes
     predictions.csv + signals.csv (or pushes the latest day's
     prediction + signal to a small key-value/Redis/DB store).
  2. The NautilusTrader Strategy itself stays thin: on_bar() reads
     "today's" already-computed prediction/signal (from the store, or
     recomputed cheaply if you keep the fitted SVR + scalers in memory
     and just .predict() on the new bar's feature vector -- no refit),
     and submits orders. This is IDENTICAL in backtest and live mode:
     in backtest, the store is the historical CSVs already produced by
     run_pipeline.py; in live, the store is updated by the same
     monthly retrain job running against the live feed.
  3. This cleanly separates "slow, monthly, CPU-heavy model fitting"
     (parallelizable across symbols, exactly as run_pipeline.py does
     with ProcessPoolExecutor) from "fast, daily, order-submission"
     logic that NautilusTrader's event loop drives -- which is the
     correct production pattern for any model that doesn't need to
     retrain on every tick.

NEXT STEP
---------------------------------------------------------------------
Tell me:
  (a) `pip show nautilus_trader` version output, and
  (b) whether you want CSV/historical bars fed in via
      `BacktestEngine.add_data` from these same project CSVs, or via
      a custom `CSVTickDataLoader`/`BarDataWrangler`,
and I will write the exact Strategy subclass + BacktestEngineConfig +
fee-model wiring against that specific API, instead of guessing at a
version and risking code that fails on import.
"""
