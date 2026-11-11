# QBEAST AI — SVR Walk-Forward Predictor + NautilusTrader Backtesting

This README tells you exactly how to lay the files out on disk, install
dependencies, and run everything **in the correct order**. Read it once
top to bottom before running anything — the pipeline has a strict
two-stage dependency: Stage 1 (prediction/signals) must finish before
Stage 2 (NautilusTrader backtests) can run.

---

## 1. Folder layout

Put everything in one project folder. It should look like this:

```
qbeast_svr_project/
│
├── requirements.txt
├── README.md                      ← this file
│
├── qbeast_slope_accel.py          ← YOUR existing modules — copy these
├── qbeast_regime_detection.py     ←   in as-is, do not edit
│
├── features.py                    ← Stage 1: feature engineering
├── svr_model.py                   ← Stage 1: walk-forward SVR
├── signals.py                     ← Stage 1: 5-class signal logic
├── metrics.py                     ← Stage 1: evaluation + plots
├── run_pipeline.py                ← Stage 1: MAIN ENTRY POINT
│
├── backtest.py                    ← Stage 2: pure-python reference backtests
├── nautilus_strategy.py           ← Stage 2: NautilusTrader v1.224.0 wiring
├── nautilus_adapter.py            ← Stage 2: design notes (read, don't run)
│
├── data/                          ← YOU create this: put symbol CSVs here
│   ├── RELIANCE.csv
│   ├── HDFCBANK.csv
│   ├── MARUTI.csv
│   └── ...
│
└── svr_results/                   ← auto-created by run_pipeline.py
    ├── RELIANCE/
    │   ├── predictions.csv
    │   ├── signals.csv
    │   ├── rolling_error.csv
    │   ├── pred_vs_actual_scatter.png
    │   ├── equity_mode1_fixed_qty.png
    │   ├── equity_mode2_capital_compounding.png
    │   ├── trades_mode1.csv
    │   ├── trades_mode2.csv
    │   └── summary.json
    ├── HDFCBANK/ ...
    ├── equity_mode3_portfolio.png
    ├── trades_mode3_portfolio.csv
    ├── mode3_summary.json
    └── overall_summary.json
```

**Rules:**
- All 11 `.py` files live flat in the project root (they `sys.path.insert`
  their own directory, so they find each other regardless of where you
  invoke `python` from — but keep them together).
- `qbeast_slope_accel.py` and `qbeast_regime_detection.py` must be your
  existing, unmodified files. `signals.py` imports them directly.
- `data/` holds raw OHLCV CSVs, one file per symbol, named exactly
  `<SYMBOL>.csv` (e.g. `RELIANCE.csv`), with columns
  `date, open, high, low, close, volume` (case-insensitive, extra
  columns are fine and ignored).
- `svr_results/` is **generated output** — never hand-edit it. Delete
  and re-run `run_pipeline.py` if you want a clean slate.

---

## 2. One-time setup

```bash
cd qbeast_svr_project
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
# (not using a venv? use: pip install -r requirements.txt --break-system-packages)
```

This installs `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `scipy`,
and `nautilus_trader==1.224.0`.

Verify:
```bash
python -c "import sklearn, nautilus_trader; print('ok')"
```

---

## 3. Stage 1 — Run the SVR walk-forward pipeline (always run this first)

This is the part that does feature engineering, monthly-retrained SVR
prediction, signal generation, metrics, and the two single-symbol
backtest modes (1 and 2) plus the portfolio mode (3), all using the
plain-Python engine in `backtest.py`. **This is also what produces
`signals.csv` per symbol, which Stage 2 (NautilusTrader) depends on.**

```bash
python run_pipeline.py \
    --symbols RELIANCE,HDFCBANK,MARUTI,ABB,BRITANNIA,BAJFINANCE,BOSCHLTD,HEROMOTOCO,MOTHERSON,INDIGO \
    --data-dir ./data \
    --out-dir ./svr_results \
    --retrain-strategy rolling \
    --capital 1000000 \
    --workers 4
```

Switch `--retrain-strategy` to `incremental` or `rolling_ewma` to compare
the three required retraining modes — just point `--out-dir` to a
different folder per strategy if you want to keep all three result sets
side by side, e.g. `./svr_results_rolling`, `./svr_results_incremental`,
`./svr_results_ewma`.

**This step is slow by design** — it refits an SVR every calendar month
per symbol across a ~7.5 year backtest window. `--workers` parallelizes
across symbols (one process per symbol); leave it at `cpu_count() - 1`
unless you're also running other heavy jobs on the machine.

Check it worked:
```bash
cat ./svr_results/overall_summary.json | python -m json.tool | head -40
ls ./svr_results/RELIANCE/
```
If any symbol shows `"status": "error"`, the `traceback` field in
`overall_summary.json` tells you why — re-run just that symbol alone to
debug (`--symbols RELIANCE`), it's faster to iterate on one at a time.

---

## 4. Stage 2 — NautilusTrader backtests (run only after Stage 1 succeeds)

`nautilus_strategy.py` reads the `signals.csv` files Stage 1 produced —
it does **not** refit the SVR itself. Run one mode at a time:

```bash
# Mode 1: fixed 100-share quantity, ₹10L allocated per stock
python nautilus_strategy.py --mode 1 --symbols RELIANCE \
    --data-dir ./data --results-dir ./svr_results --capital 1000000

# Mode 2: capital-compounding, single stock
python nautilus_strategy.py --mode 2 --symbols RELIANCE \
    --data-dir ./data --results-dir ./svr_results --capital 1000000

# Mode 3: portfolio, equal-weighted across all listed symbols
python nautilus_strategy.py --mode 3 \
    --symbols RELIANCE,HDFCBANK,MARUTI,ABB,BRITANNIA,BAJFINANCE,BOSCHLTD,HEROMOTOCO,MOTHERSON,INDIGO \
    --data-dir ./data --results-dir ./svr_results --capital 1000000
```

**Before trusting these numbers:** this file was written against the
documented v1.224.0 API but not import-tested in my environment (no
`nautilus_trader` installed here). Run Mode 1 on a single symbol first;
if anything throws a `TypeError`/`AttributeError` on construction, it's
almost certainly in `make_equity_instrument()` or `DAILY_BAR_SPEC` near
the top of the file (instrument/bar-spec constructor kwargs are the
parts most likely to have drifted slightly in your exact build) — fix
those two spots, the rest of the control flow is stable.

If you'd rather skip NautilusTrader entirely and trust the pure-Python
engine, **Stage 1 already gives you all three modes** via `backtest.py`
— that's what produces `equity_mode1/2/3...png` and `trades_mode*.csv`
inside `svr_results/`. NautilusTrader is only needed if you specifically
require its execution/portfolio/reporting machinery for downstream live
deployment.

---

## 5. What to look at, and in what order

1. `svr_results/<SYMBOL>/pred_vs_actual_scatter.png` — is the SVR any
   good for this symbol? Points hugging the red 45° line = good.
2. `svr_results/<SYMBOL>/summary.json` → `metrics` block — MAE, RMSE,
   MAPE, R², Directional Accuracy, Hit Ratio, Correlation.
3. `svr_results/<SYMBOL>/signals.csv` — eyeball the signal distribution
   (`value_counts()` on the `signal` column) to sanity-check it isn't
   stuck on Hold or flipping every day.
4. `svr_results/<SYMBOL>/equity_mode1_fixed_qty.png` and
   `equity_mode2_capital_compounding.png`, then
   `svr_results/equity_mode3_portfolio.png`.
5. `svr_results/<SYMBOL>/trades_mode1.csv` / `trades_mode2.csv` and
   `svr_results/trades_mode3_portfolio.csv` for trade-level detail
   (entry/exit dates, prices, gross vs net P&L after the Indian
   delivery costs).

---

## 6. Re-running after changes

- Changed a feature/signal threshold? Just re-run `run_pipeline.py` —
  it overwrites `predictions.csv`/`signals.csv` per symbol; nothing
  needs to be deleted manually.
- Switching retrain strategy or symbol list? Point `--out-dir` to a
  fresh folder so you don't mix result sets from different runs.

---

## 7. File responsibility quick-reference

| File | Stage | What it does | You edit it? |
|---|---|---|---|
| `qbeast_slope_accel.py` | 1 | your existing slope/accel/signal module | no, as-is |
| `qbeast_regime_detection.py` | 1 | your existing regime detector | no, as-is |
| `features.py` | 1 | causal technical features | maybe — add indicators |
| `svr_model.py` | 1 | walk-forward SVR, 3 retrain strategies, calibration | maybe — tune `SVRConfig` |
| `signals.py` | 1 | 5-class Buy/Sell signal logic | maybe — tune thresholds |
| `metrics.py` | 1 | evaluation metrics + plots | rarely |
| `run_pipeline.py` | 1 | **orchestrator / main entry point** | CLI args only |
| `backtest.py` | 2 | pure-python reference backtest (all 3 modes) | rarely |
| `nautilus_strategy.py` | 2 | NautilusTrader v1.224.0 engine/strategies | yes — verify against your build |
| `nautilus_adapter.py` | 2 | design notes only, not runnable | n/a |
| `requirements.txt` | — | pinned dependencies | only to bump versions |
