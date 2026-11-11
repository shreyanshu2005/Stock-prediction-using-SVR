#!/usr/bin/env python3
"""
qbeast_svr_dashboard.py
=====================================================================
Results dashboard for the QBEAST SVR walk-forward pipeline
(run_pipeline.py output). Run this AFTER run_pipeline.py has produced
a results folder, and it will auto-discover every symbol and show:

  - Cross-symbol leaderboard (MAE / RMSE / MAPE / R2 / DirAcc / HitRatio)
  - Predicted vs Actual 45-degree scatter (interactive, built live from
    predictions.csv -- not just the static PNG)
  - Rolling MAE / RMSE
  - Signal distribution (Strong Buy / Buy / Hold / Sell / Strong Sell)
  - Equity curves for Mode 1 (fixed qty) and Mode 2 (capital compounding),
    rebuilt live from trades_mode1.csv / trades_mode2.csv + predictions.csv
  - Trade tables (Mode 1 / Mode 2) with gross vs net P&L
  - Portfolio Mode 3 equity curve + trades, across the whole universe
  - Raw summary.json / overall_summary.json viewer

USAGE
-----
    pip install dash plotly pandas --break-system-packages
    python qbeast_svr_dashboard.py --results-dir ./svr_results --port 8060

Or set the env var QBEAST_RESULTS_DIR instead of --results-dir.
The folder must look like what run_pipeline.py produces:

    svr_results/
      <SYMBOL>/
        predictions.csv
        rolling_error.csv
        signals.csv
        trades_mode1.csv
        trades_mode2.csv
        summary.json
        pred_vs_actual_scatter.png      (optional, shown as fallback)
        equity_mode1_fixed_qty.png      (optional, shown as fallback)
        equity_mode2_capital_compounding.png (optional)
      overall_summary.json
      mode3_summary.json                (optional)
      trades_mode3_portfolio.csv        (optional)
"""
from __future__ import annotations

import os
import json
import argparse
import base64
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, dash_table

# ══════════════════════════════════════════════════════════════════
# CONFIG / CLI
# ══════════════════════════════════════════════════════════════════
ap = argparse.ArgumentParser()
ap.add_argument("--results-dir", default=os.environ.get("QBEAST_RESULTS_DIR", "./svr_results"))
ap.add_argument("--port", type=int, default=8060)
args, _unknown = ap.parse_known_args()
RESULTS_DIR = os.path.abspath(args.results_dir)

SIGNAL_LABELS = {2: "Strong Buy", 1: "Buy", 0: "Hold", -1: "Sell", -2: "Strong Sell"}
SIGNAL_COLORS = {"Strong Buy": "#00C853", "Buy": "#69F0AE", "Hold": "#90A4AE",
                 "Sell": "#FFAB91", "Strong Sell": "#FF1744"}

C = dict(bg="#0d1117", paper="#0d1117", grid="#1e2a35", font="#e8eaed", sub="#90a4ae",
         bull="#00FF94", bear="#FF0004", blue="#42A5F5", gold="#FFD700",
         purple="#CE93D8", teal="#80CBC4", orange="#FF9800")

TEMPLATE = "plotly_dark"


# ══════════════════════════════════════════════════════════════════
# DISCOVERY / LOADERS  (all defensive -- a partially-run pipeline,
# missing files, or a still-in-progress symbol must never crash the UI)
# ══════════════════════════════════════════════════════════════════
def discover_symbols(results_dir: str) -> list[str]:
    if not os.path.isdir(results_dir):
        return []
    syms = []
    for name in sorted(os.listdir(results_dir)):
        p = os.path.join(results_dir, name)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "predictions.csv")):
            syms.append(name)
    return syms


def _read_csv_safe(path, **kw):
    try:
        if os.path.exists(path):
            return pd.read_csv(path, **kw)
    except Exception:
        pass
    return None


def _read_json_safe(path):
    try:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    except Exception:
        pass
    return None


def load_symbol_bundle(sym: str) -> dict:
    base = os.path.join(RESULTS_DIR, sym)
    pred = _read_csv_safe(os.path.join(base, "predictions.csv"), parse_dates=["date"], index_col="date")
    roll = _read_csv_safe(os.path.join(base, "rolling_error.csv"), parse_dates=[0], index_col=0)
    sig = _read_csv_safe(os.path.join(base, "signals.csv"), parse_dates=[0], index_col=0)
    t1 = _read_csv_safe(os.path.join(base, "trades_mode1.csv"))
    t2 = _read_csv_safe(os.path.join(base, "trades_mode2.csv"))
    summary = _read_json_safe(os.path.join(base, "summary.json"))
    return dict(base=base, pred=pred, roll=roll, sig=sig, t1=t1, t2=t2, summary=summary)


def load_overall():
    overall = _read_json_safe(os.path.join(RESULTS_DIR, "overall_summary.json")) or {}
    mode3 = _read_json_safe(os.path.join(RESULTS_DIR, "mode3_summary.json"))
    t3 = _read_csv_safe(os.path.join(RESULTS_DIR, "trades_mode3_portfolio.csv"))
    return overall, mode3, t3


def leaderboard_df(overall: dict) -> pd.DataFrame:
    rows = []
    for sym, r in overall.items():
        if not isinstance(r, dict):
            continue
        status = r.get("status", "?")
        m = r.get("metrics") or {}
        m1 = r.get("mode1") or {}
        m2 = r.get("mode2") or {}
        rows.append(dict(
            Symbol=sym, Status=status,
            MAE=m.get("MAE"), RMSE=m.get("RMSE"), MAPE=m.get("MAPE"),
            R2=m.get("R2"), Corr=m.get("Correlation"),
            DirAcc=m.get("DirectionalAccuracy"), HitRatio=m.get("HitRatio"),
            N=m.get("N"),
            Mode1_NetPnL=m1.get("net_pnl"), Mode1_Ret_pct=m1.get("total_return_pct"),
            Mode1_Trades=m1.get("n_trades"),
            Mode2_NetPnL=m2.get("net_pnl"), Mode2_Ret_pct=m2.get("total_return_pct"),
            Mode2_Trades=m2.get("n_trades"),
        ))
    df = pd.DataFrame(rows)
    if not df.empty:
        for c in df.columns:
            if c not in ("Symbol", "Status"):
                df[c] = pd.to_numeric(df[c], errors="coerce").round(3)
    return df


def reconstruct_equity(pred: pd.DataFrame, trades: pd.DataFrame | None,
                       capital: float = 1_000_000.0) -> pd.Series:
    """Rebuild a mark-to-market equity curve from a trade list + the
    close price series in predictions.csv, since the pipeline only saves
    the trade ledger (not the daily curve) to disk per symbol."""
    if pred is None or pred.empty:
        return pd.Series(dtype=float)
    px = pred["close_today"] if "close_today" in pred.columns else None
    if px is None:
        return pd.Series(dtype=float)
    px = px.sort_index()
    if trades is None or trades.empty:
        return pd.Series(capital, index=px.index)

    trades = trades.copy()
    trades["entry_date"] = pd.to_datetime(trades["entry_date"])
    trades["exit_date"] = pd.to_datetime(trades["exit_date"])

    cash = capital
    position_qty = 0
    eq = []
    open_trades = trades.sort_values("entry_date").to_dict("records")
    ti = 0
    active = None
    realized_cash = capital
    for date in px.index:
        # process exits / entries that happen on this date in trade order
        while ti < len(open_trades) and open_trades[ti]["entry_date"] <= date and active is None:
            active = open_trades[ti]
            ti += 1
        if active is not None and active["exit_date"] == date:
            realized_cash += active["net_pnl"] if not pd.isna(active["net_pnl"]) else 0.0
            active = None
        mtm = realized_cash
        if active is not None:
            qty = active.get("qty", 0) or 0
            mtm = realized_cash + qty * (px.loc[date] - active["entry_price"])
        eq.append(mtm)
    return pd.Series(eq, index=px.index)


def png_to_data_uri(path: str) -> str | None:
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:image/png;base64,{b64}"


# ══════════════════════════════════════════════════════════════════
# FIGURE BUILDERS
# ══════════════════════════════════════════════════════════════════
def empty_fig(msg="No data"):
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font=dict(size=16, color=C["sub"]))
    fig.update_layout(template=TEMPLATE, paper_bgcolor=C["paper"], plot_bgcolor=C["bg"],
                      height=300, xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


def fig_scatter_45(pred: pd.DataFrame, sym: str):
    if pred is None or pred.empty:
        return empty_fig("No predictions.csv")
    df = pred.dropna(subset=["final_pred", "actual_next_close"])
    if df.empty:
        return empty_fig("No completed predictions yet")
    x = df["actual_next_close"].values
    y = df["final_pred"].values
    lo, hi = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
    ss_res = float(np.sum((y - x) ** 2))
    ss_tot = float(np.sum((x - x.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Predictions",
                             marker=dict(size=5, color="#16c79a", opacity=0.55,
                                        line=dict(width=0.3, color="#0c8f6e")),
                             hovertemplate="Actual: %{x:.2f}<br>Pred: %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect 45°",
                             line=dict(color="#FF3B3B", width=1.6, dash="dash")))
    if len(x) > 1:
        b, a = np.polyfit(x, y, 1)
        fig.add_trace(go.Scatter(x=[lo, hi], y=[b * lo + a, b * hi + a], mode="lines",
                                 name=f"Fit (R²={r2:.3f})", line=dict(color=C["gold"], width=2)))
    fig.update_layout(template=TEMPLATE, paper_bgcolor=C["paper"], plot_bgcolor=C["bg"],
                      title=f"{sym} — Predicted vs Actual (1-day-ahead)",
                      xaxis_title="Actual next-day close", yaxis_title="Predicted next-day close",
                      height=480, legend=dict(orientation="h", y=1.08),
                      font=dict(color=C["font"]))
    fig.update_xaxes(gridcolor=C["grid"])
    fig.update_yaxes(gridcolor=C["grid"])
    return fig


def fig_rolling_error(roll: pd.DataFrame, sym: str):
    if roll is None or roll.empty:
        return empty_fig("No rolling_error.csv")
    fig = go.Figure()
    if "rolling_mae" in roll.columns:
        fig.add_trace(go.Scatter(x=roll.index, y=roll["rolling_mae"], name="Rolling MAE",
                                 line=dict(color=C["blue"], width=1.6)))
    if "rolling_rmse" in roll.columns:
        fig.add_trace(go.Scatter(x=roll.index, y=roll["rolling_rmse"], name="Rolling RMSE",
                                 line=dict(color=C["orange"], width=1.6, dash="dot")))
    fig.update_layout(template=TEMPLATE, paper_bgcolor=C["paper"], plot_bgcolor=C["bg"],
                      title=f"{sym} — Rolling Prediction Error (20d)", height=380,
                      font=dict(color=C["font"]), legend=dict(orientation="h", y=1.1))
    fig.update_xaxes(gridcolor=C["grid"])
    fig.update_yaxes(gridcolor=C["grid"], title="₹ error")
    return fig


def fig_pred_vs_actual_series(pred: pd.DataFrame, sym: str):
    if pred is None or pred.empty:
        return empty_fig("No predictions.csv")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pred.index, y=pred["close_today"], name="Close (today)",
                             line=dict(color=C["blue"], width=1.4)))
    fig.add_trace(go.Scatter(x=pred.index, y=pred["final_pred"], name="Predicted next close",
                             line=dict(color=C["gold"], width=1.2, dash="dot")))
    fig.update_layout(template=TEMPLATE, paper_bgcolor=C["paper"], plot_bgcolor=C["bg"],
                      title=f"{sym} — Close vs Predicted Next-Day Close", height=420,
                      font=dict(color=C["font"]), legend=dict(orientation="h", y=1.08))
    fig.update_xaxes(gridcolor=C["grid"])
    fig.update_yaxes(gridcolor=C["grid"], title="₹")
    return fig


def fig_signal_dist(sig: pd.DataFrame, sym: str):
    if sig is None or sig.empty or "signal" not in sig.columns:
        return empty_fig("No signals.csv")
    labels = sig["signal"].map(SIGNAL_LABELS).fillna("Hold")
    counts = labels.value_counts().reindex(
        ["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"]).fillna(0)
    fig = go.Figure(go.Bar(x=counts.index, y=counts.values,
                           marker=dict(color=[SIGNAL_COLORS[k] for k in counts.index])))
    fig.update_layout(template=TEMPLATE, paper_bgcolor=C["paper"], plot_bgcolor=C["bg"],
                      title=f"{sym} — Signal Distribution", height=360,
                      font=dict(color=C["font"]))
    fig.update_yaxes(gridcolor=C["grid"], title="Days")
    return fig


def fig_signal_timeline(sig: pd.DataFrame, sym: str):
    if sig is None or sig.empty or "signal" not in sig.columns or "close" not in sig.columns:
        return empty_fig("No signals.csv")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sig.index, y=sig["close"], name="Close",
                             line=dict(color=C["sub"], width=1.1)))
    for lab, color in SIGNAL_COLORS.items():
        code = [k for k, v in SIGNAL_LABELS.items() if v == lab][0]
        if code == 0:
            continue
        m = sig["signal"] == code
        if m.any():
            fig.add_trace(go.Scatter(x=sig.index[m], y=sig.loc[m, "close"], mode="markers",
                                     name=lab, marker=dict(size=7, color=color,
                                                           symbol="triangle-up" if code > 0 else "triangle-down")))
    fig.update_layout(template=TEMPLATE, paper_bgcolor=C["paper"], plot_bgcolor=C["bg"],
                      title=f"{sym} — Buy/Sell Signals on Price", height=460,
                      font=dict(color=C["font"]), legend=dict(orientation="h", y=1.08))
    fig.update_xaxes(gridcolor=C["grid"])
    fig.update_yaxes(gridcolor=C["grid"], title="₹")
    return fig


def fig_equity(eq: pd.Series, title: str, color=C["bull"]):
    if eq is None or eq.empty:
        return empty_fig("No trade data")
    fig = go.Figure(go.Scatter(x=eq.index, y=eq.values, line=dict(color=color, width=1.6)))
    fig.update_layout(template=TEMPLATE, paper_bgcolor=C["paper"], plot_bgcolor=C["bg"],
                      title=title, height=380, font=dict(color=C["font"]))
    fig.update_xaxes(gridcolor=C["grid"])
    fig.update_yaxes(gridcolor=C["grid"], title="₹")
    return fig


def fig_leaderboard_bar(df: pd.DataFrame, col: str, title: str, ascending=False):
    if df.empty or col not in df.columns:
        return empty_fig("No overall_summary.json")
    d = df.dropna(subset=[col]).sort_values(col, ascending=ascending).head(30)
    if d.empty:
        return empty_fig("No data for " + col)
    colors = [C["bull"] if v >= 0 else C["bear"] for v in d[col]] if "Ret" in col or "PnL" in col else C["blue"]
    fig = go.Figure(go.Bar(x=d["Symbol"], y=d[col], marker=dict(color=colors)))
    fig.update_layout(template=TEMPLATE, paper_bgcolor=C["paper"], plot_bgcolor=C["bg"],
                      title=title, height=380, font=dict(color=C["font"]))
    fig.update_xaxes(gridcolor=C["grid"], tickangle=-45)
    fig.update_yaxes(gridcolor=C["grid"])
    return fig


# ══════════════════════════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════════════════════════
app = dash.Dash(__name__, title="QBEAST SVR — Results Dashboard",
                suppress_callback_exceptions=True)

DARK_CSS = f"""
* {{ box-sizing:border-box; margin:0; padding:0; }}
body {{ background:{C['bg']}; color:{C['font']}; font-family:'JetBrains Mono','Fira Code','Courier New',monospace; }}
.tab-bar .tab {{ background:#161b22 !important; color:{C['sub']} !important; border:1px solid #263238 !important; }}
.tab-bar .tab--selected {{ background:#1565C0 !important; color:white !important; }}
table {{ color:{C['font']}; }}
::-webkit-scrollbar {{ width:6px; height:6px; }}
::-webkit-scrollbar-thumb {{ background:#37474f; border-radius:3px; }}
"""
app.index_string = f"""<!DOCTYPE html><html><head>{{%metas%}}<title>{{%title%}}</title>
{{%favicon%}}{{%css%}}<style>{DARK_CSS}</style></head><body>{{%app_entry%}}
<footer>{{%config%}}{{%scripts%}}{{%renderer%}}</footer></body></html>"""

HDR = {"display": "flex", "alignItems": "center", "gap": "14px", "background": "#0a0f16",
       "padding": "10px 16px", "borderBottom": "1px solid #263238", "flexWrap": "wrap"}

KPI_STYLE = {"background": "#161b22", "border": "1px solid #263238", "borderRadius": "8px",
            "padding": "10px 16px", "minWidth": "120px"}


def kpi(label, val_id):
    return html.Div(style=KPI_STYLE, children=[
        html.Div(label, style={"color": C["sub"], "fontSize": "10px", "textTransform": "uppercase"}),
        html.Div(id=val_id, style={"color": C["font"], "fontSize": "18px", "fontWeight": "bold"}),
    ])


app.layout = html.Div(style={"background": C["bg"], "minHeight": "100vh"}, children=[

    html.Div(style=HDR, children=[
        html.Div("QBEAST SVR — Results Dashboard", style={
            "color": C["orange"], "fontWeight": "bold", "fontSize": "16px"}),
        html.Div(f"Results dir: {RESULTS_DIR}", style={"color": C["sub"], "fontSize": "11px"}),
        html.Button("⟳ Refresh", id="btn-refresh", n_clicks=0, style={
            "marginLeft": "auto", "background": "#1565C0", "color": "white", "border": "none",
            "borderRadius": "5px", "padding": "6px 14px", "cursor": "pointer", "fontFamily": "inherit"}),
    ]),

    dcc.Store(id="store-symbols"),

    dcc.Tabs(id="tabs", value="tab-overview", className="tab-bar", children=[
        dcc.Tab(label="Overview / Leaderboard", value="tab-overview"),
        dcc.Tab(label="Symbol Detail", value="tab-symbol"),
        dcc.Tab(label="Portfolio (Mode 3)", value="tab-portfolio"),
        dcc.Tab(label="Raw Summaries", value="tab-raw"),
    ]),

    html.Div(id="tab-content", style={"padding": "14px"}),

    dcc.Interval(id="iv-init", interval=500, n_intervals=0, max_intervals=1),
])


# ── Tab: Overview ────────────────────────────────────────────────
def render_overview():
    overall, mode3, t3 = load_overall()
    df = leaderboard_df(overall)
    if df.empty:
        return html.Div([
            html.Div(f"No overall_summary.json found yet in '{RESULTS_DIR}'.",
                     style={"color": C["sub"], "padding": "30px", "fontSize": "13px"}),
            html.Div("Run run_pipeline.py first, then click Refresh.",
                     style={"color": C["sub"], "padding": "0 30px", "fontSize": "12px"}),
        ])

    n_ok = int((df["Status"] == "ok").sum())
    n_err = int((df["Status"] == "error").sum())
    avg_dir = df["DirAcc"].mean()
    avg_r2 = df["R2"].mean()

    kpis = html.Div(style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "16px"}, children=[
        kpi_static("Symbols", str(len(df))),
        kpi_static("OK", str(n_ok), color=C["bull"]),
        kpi_static("Errors", str(n_err), color=C["bear"] if n_err else C["sub"]),
        kpi_static("Avg Dir. Accuracy", f"{avg_dir:.1f}%" if pd.notna(avg_dir) else "—"),
        kpi_static("Avg R²", f"{avg_r2:.3f}" if pd.notna(avg_r2) else "—"),
    ])

    table = dash_table.DataTable(
        data=df.to_dict("records"), columns=[{"name": c, "id": c} for c in df.columns],
        sort_action="native", filter_action="native", page_size=20,
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": "#161b22", "color": C["font"], "fontWeight": "bold",
                     "border": "1px solid #263238"},
        style_cell={"backgroundColor": "#0d1117", "color": C["font"], "border": "1px solid #1e2a35",
                   "fontFamily": "inherit", "fontSize": "11px", "padding": "5px"},
        style_data_conditional=[
            {"if": {"filter_query": "{Status} = error"}, "backgroundColor": "#2b0f12"},
            {"if": {"filter_query": "{Mode1_Ret_pct} > 0", "column_id": "Mode1_Ret_pct"}, "color": C["bull"]},
            {"if": {"filter_query": "{Mode1_Ret_pct} < 0", "column_id": "Mode1_Ret_pct"}, "color": C["bear"]},
            {"if": {"filter_query": "{Mode2_Ret_pct} > 0", "column_id": "Mode2_Ret_pct"}, "color": C["bull"]},
            {"if": {"filter_query": "{Mode2_Ret_pct} < 0", "column_id": "Mode2_Ret_pct"}, "color": C["bear"]},
        ],
    )

    charts = html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "10px",
                             "marginTop": "16px"}, children=[
        dcc.Graph(figure=fig_leaderboard_bar(df, "Mode1_Ret_pct", "Mode 1 Total Return % by Symbol")),
        dcc.Graph(figure=fig_leaderboard_bar(df, "Mode2_Ret_pct", "Mode 2 Total Return % by Symbol")),
        dcc.Graph(figure=fig_leaderboard_bar(df, "DirAcc", "Directional Accuracy % by Symbol")),
        dcc.Graph(figure=fig_leaderboard_bar(df, "R2", "R² by Symbol")),
    ])

    return html.Div([kpis, table, charts])


def kpi_static(label, val, color=None):
    return html.Div(style=KPI_STYLE, children=[
        html.Div(label, style={"color": C["sub"], "fontSize": "10px", "textTransform": "uppercase"}),
        html.Div(val, style={"color": color or C["font"], "fontSize": "18px", "fontWeight": "bold"}),
    ])


# ── Tab: Symbol detail ───────────────────────────────────────────
def render_symbol_tab(selected_sym, symbols):
    if not symbols:
        return html.Div(f"No symbol folders with predictions.csv found in '{RESULTS_DIR}'.",
                        style={"color": C["sub"], "padding": "30px"})
    sym = selected_sym if selected_sym in symbols else symbols[0]
    bundle = load_symbol_bundle(sym)
    pred, roll, sig, t1, t2, summary = (bundle["pred"], bundle["roll"], bundle["sig"],
                                        bundle["t1"], bundle["t2"], bundle["summary"])

    m = (summary or {}).get("metrics", {})
    m1 = (summary or {}).get("mode1", {})
    m2 = (summary or {}).get("mode2", {})

    kpis = html.Div(style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"}, children=[
        kpi_static("MAE", f"{m.get('MAE', float('nan')):.3f}" if m.get("MAE") is not None else "—"),
        kpi_static("RMSE", f"{m.get('RMSE', float('nan')):.3f}" if m.get("RMSE") is not None else "—"),
        kpi_static("MAPE %", f"{m.get('MAPE', float('nan')):.2f}" if m.get("MAPE") is not None else "—"),
        kpi_static("R²", f"{m.get('R2', float('nan')):.3f}" if m.get("R2") is not None else "—"),
        kpi_static("Dir. Acc %", f"{m.get('DirectionalAccuracy', float('nan')):.1f}"
                   if m.get("DirectionalAccuracy") is not None else "—"),
        kpi_static("Hit Ratio %", f"{m.get('HitRatio', float('nan')):.1f}"
                   if m.get("HitRatio") is not None else "—"),
        kpi_static("Mode1 Net P&L ₹", f"{m1.get('net_pnl', 0):,.0f}" if m1 else "—",
                  color=(C["bull"] if (m1.get("net_pnl") or 0) >= 0 else C["bear"]) if m1 else None),
        kpi_static("Mode2 Net P&L ₹", f"{m2.get('net_pnl', 0):,.0f}" if m2 else "—",
                  color=(C["bull"] if (m2.get("net_pnl") or 0) >= 0 else C["bear"]) if m2 else None),
    ])

    eq1 = reconstruct_equity(pred, t1, capital=1_000_000.0)
    eq2 = reconstruct_equity(pred, t2, capital=1_000_000.0)

    def trade_table(df, title):
        if df is None or df.empty:
            return html.Div([html.H4(title, style={"color": C["sub"], "fontSize": "12px"}),
                             html.Div("No trades", style={"color": C["sub"], "fontSize": "11px"})])
        d = df.copy().round(2)
        return html.Div([
            html.H4(title, style={"color": C["sub"], "fontSize": "12px", "margin": "8px 0 4px"}),
            dash_table.DataTable(
                data=d.to_dict("records"), columns=[{"name": c, "id": c} for c in d.columns],
                page_size=10, sort_action="native", style_table={"overflowX": "auto"},
                style_header={"backgroundColor": "#161b22", "color": C["font"], "border": "1px solid #263238"},
                style_cell={"backgroundColor": "#0d1117", "color": C["font"], "border": "1px solid #1e2a35",
                           "fontFamily": "inherit", "fontSize": "10px", "padding": "4px"},
                style_data_conditional=[
                    {"if": {"filter_query": "{net_pnl} > 0", "column_id": "net_pnl"}, "color": C["bull"]},
                    {"if": {"filter_query": "{net_pnl} < 0", "column_id": "net_pnl"}, "color": C["bear"]},
                ],
            )])

    return html.Div([
        html.Div(style={"display": "flex", "gap": "10px", "alignItems": "center", "marginBottom": "10px"}, children=[
            html.Div("Symbol", style={"color": C["sub"], "fontSize": "11px"}),
            dcc.Dropdown(id="dd-symbol", options=[{"label": s, "value": s} for s in symbols],
                        value=sym, clearable=False, style={"width": "220px", "color": "#000"}),
        ]),
        kpis,
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "10px"}, children=[
            dcc.Graph(figure=fig_scatter_45(pred, sym)),
            dcc.Graph(figure=fig_rolling_error(roll, sym)),
        ]),
        dcc.Graph(figure=fig_pred_vs_actual_series(pred, sym)),
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "10px"}, children=[
            dcc.Graph(figure=fig_signal_dist(sig, sym)),
            dcc.Graph(figure=fig_signal_timeline(sig, sym)),
        ]),
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "10px"}, children=[
            dcc.Graph(figure=fig_equity(eq1, f"{sym} — Mode 1 Equity (Fixed 100-share Qty)", C["bull"])),
            dcc.Graph(figure=fig_equity(eq2, f"{sym} — Mode 2 Equity (Capital Compounding)", C["blue"])),
        ]),
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "10px"}, children=[
            trade_table(t1, "Mode 1 trades"), trade_table(t2, "Mode 2 trades"),
        ]),
    ])


# ── Tab: Portfolio (mode 3) ──────────────────────────────────────
def render_portfolio_tab():
    overall, mode3, t3 = load_overall()
    if not mode3:
        return html.Div("No mode3_summary.json found (Mode 3 portfolio backtest hasn't been run / saved yet).",
                        style={"color": C["sub"], "padding": "30px"})

    kpis = html.Div(style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"}, children=[
        kpi_static("Net P&L ₹", f"{mode3.get('net_pnl', 0):,.0f}",
                  color=C["bull"] if (mode3.get("net_pnl") or 0) >= 0 else C["bear"]),
        kpi_static("Gross P&L ₹", f"{mode3.get('gross_pnl', 0):,.0f}"),
        kpi_static("Total Return %", f"{mode3.get('total_return_pct', 0):.2f}",
                  color=C["bull"] if (mode3.get("total_return_pct") or 0) >= 0 else C["bear"]),
        kpi_static("Final Equity ₹", f"{mode3.get('final_equity', 0):,.0f}"),
        kpi_static("Trades", str(mode3.get("n_trades", "—"))),
    ])

    eq_fig = empty_fig("No per-date equity stored — see trades table below")
    table = html.Div("No trades_mode3_portfolio.csv found.", style={"color": C["sub"]})
    if t3 is not None and not t3.empty:
        d = t3.copy()
        for c in ("entry_date", "exit_date"):
            if c in d.columns:
                d[c] = pd.to_datetime(d[c], errors="coerce")
        d = d.sort_values("exit_date")
        if "net_pnl" in d.columns:
            cum = d["net_pnl"].fillna(0).cumsum() + 1_000_000.0
            eq_fig = go.Figure(go.Scatter(x=d["exit_date"], y=cum, line=dict(color=C["bull"], width=1.6)))
            eq_fig.update_layout(template=TEMPLATE, paper_bgcolor=C["paper"], plot_bgcolor=C["bg"],
                                 title="Mode 3 — Portfolio Equity (cumulative realized P&L)",
                                 height=400, font=dict(color=C["font"]))
            eq_fig.update_xaxes(gridcolor=C["grid"])
            eq_fig.update_yaxes(gridcolor=C["grid"], title="₹")

        by_sym = d.groupby("symbol")["net_pnl"].sum().sort_values(ascending=False).reset_index() \
            if "symbol" in d.columns else pd.DataFrame()
        contrib_fig = empty_fig("No symbol column in trades")
        if not by_sym.empty:
            colors = [C["bull"] if v >= 0 else C["bear"] for v in by_sym["net_pnl"]]
            contrib_fig = go.Figure(go.Bar(x=by_sym["symbol"], y=by_sym["net_pnl"], marker=dict(color=colors)))
            contrib_fig.update_layout(template=TEMPLATE, paper_bgcolor=C["paper"], plot_bgcolor=C["bg"],
                                      title="Mode 3 — Net P&L Contribution by Symbol", height=380,
                                      font=dict(color=C["font"]))
            contrib_fig.update_xaxes(gridcolor=C["grid"], tickangle=-45)
            contrib_fig.update_yaxes(gridcolor=C["grid"])

        dd = d.round(2)
        table = dash_table.DataTable(
            data=dd.to_dict("records"), columns=[{"name": c, "id": c} for c in dd.columns],
            page_size=15, sort_action="native", filter_action="native",
            style_table={"overflowX": "auto"},
            style_header={"backgroundColor": "#161b22", "color": C["font"], "border": "1px solid #263238"},
            style_cell={"backgroundColor": "#0d1117", "color": C["font"], "border": "1px solid #1e2a35",
                       "fontFamily": "inherit", "fontSize": "10px", "padding": "4px"},
            style_data_conditional=[
                {"if": {"filter_query": "{net_pnl} > 0", "column_id": "net_pnl"}, "color": C["bull"]},
                {"if": {"filter_query": "{net_pnl} < 0", "column_id": "net_pnl"}, "color": C["bear"]},
            ],
        )
        return html.Div([kpis, dcc.Graph(figure=eq_fig), dcc.Graph(figure=contrib_fig), table])

    return html.Div([kpis, dcc.Graph(figure=eq_fig), table])


# ── Tab: Raw summaries ───────────────────────────────────────────
def render_raw_tab(symbols):
    overall, mode3, _ = load_overall()
    blocks = [
        html.H4("overall_summary.json", style={"color": C["sub"], "fontSize": "12px"}),
        html.Pre(json.dumps(overall, indent=2, default=str)[:20000],
                 style={"background": "#161b22", "padding": "10px", "borderRadius": "6px",
                       "fontSize": "10px", "overflowX": "auto", "maxHeight": "400px"}),
    ]
    if mode3:
        blocks += [
            html.H4("mode3_summary.json", style={"color": C["sub"], "fontSize": "12px"}),
            html.Pre(json.dumps(mode3, indent=2, default=str),
                     style={"background": "#161b22", "padding": "10px", "borderRadius": "6px",
                           "fontSize": "10px", "overflowX": "auto"}),
        ]
    blocks += [
        html.H4("Discovered symbols", style={"color": C["sub"], "fontSize": "12px"}),
        html.Div(", ".join(symbols) if symbols else "(none)",
                 style={"fontSize": "11px", "color": C["font"]}),
    ]
    return html.Div(blocks)


# ══════════════════════════════════════════════════════════════════
# CALLBACKS
# ══════════════════════════════════════════════════════════════════
@app.callback(Output("store-symbols", "data"),
             Input("iv-init", "n_intervals"), Input("btn-refresh", "n_clicks"))
def refresh_symbols(_n, _clicks):
    return discover_symbols(RESULTS_DIR)


@app.callback(Output("tab-content", "children"),
             Input("tabs", "value"), Input("store-symbols", "data"),
             State("dd-symbol", "value") if False else Input("tabs", "value"))
def render_tab(tab, symbols, _dummy):
    symbols = symbols or discover_symbols(RESULTS_DIR)
    if tab == "tab-overview":
        return render_overview()
    if tab == "tab-symbol":
        return render_symbol_tab(symbols[0] if symbols else None, symbols)
    if tab == "tab-portfolio":
        return render_portfolio_tab()
    if tab == "tab-raw":
        return render_raw_tab(symbols)
    return html.Div()


@app.callback(Output("tab-content", "children", allow_duplicate=True),
             Input("dd-symbol", "value"), State("store-symbols", "data"),
             prevent_initial_call=True)
def on_symbol_change(sym, symbols):
    symbols = symbols or discover_symbols(RESULTS_DIR)
    return render_symbol_tab(sym, symbols)


if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  QBEAST SVR — Results Dashboard")
    print(f"  Results dir : {RESULTS_DIR}")
    syms = discover_symbols(RESULTS_DIR)
    print(f"  Symbols found: {len(syms)}  -> {syms[:10]}{' ...' if len(syms) > 10 else ''}")
    print(f"  URL: http://127.0.0.1:{args.port}")
    print(f"{'='*60}\n")
    app.run(host="0.0.0.0", debug=False, port=args.port)