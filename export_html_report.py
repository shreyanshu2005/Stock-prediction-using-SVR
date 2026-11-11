#!/usr/bin/env python3
"""
export_html_report.py
=====================================================================
Generate a single, fully self-contained HTML report from the
svr_results/ folder produced by run_pipeline.py.

The output file has ZERO external dependencies — one .html file you
can email to your supervisor, open in any browser, and share without
needing Python, Dash, or an internet connection.

USAGE
-----
    python export_html_report.py --results-dir ./svr_results --out qbeast_report.html

All Plotly charts are embedded as JSON (plotly.js is loaded from CDN
once — the only network call when the page opens; it works offline if
your browser has cached plotly.js from any previous visit).
"""
from __future__ import annotations

import os, sys, json, argparse, textwrap
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# ── CLI ─────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--results-dir", default=os.environ.get("QBEAST_RESULTS_DIR", "./svr_results"))
ap.add_argument("--out", default="qbeast_report.html")
ap.add_argument("--capital", type=float, default=1_000_000.0,
                help="Starting capital used in the backtest (default 1,000,000)")
args, _ = ap.parse_known_args()
RESULTS_DIR = os.path.abspath(args.results_dir)
OUT_PATH    = args.out
CAPITAL     = args.capital

# ── Palette ──────────────────────────────────────────────────────
BG      = "#0d1117"
PAPER   = "#0d1117"
GRID    = "#1e2a35"
FONT    = "#e8eaed"
SUB     = "#90a4ae"
BULL    = "#00FF94"
BEAR    = "#FF4444"
BLUE    = "#42A5F5"
GOLD    = "#FFD700"
ORANGE  = "#FF9800"
PURPLE  = "#CE93D8"
TEAL    = "#80CBC4"
CARD_BG = "#161b22"
BORDER  = "#263238"
TPL     = "plotly_dark"

SIGNAL_LABELS = {2: "Strong Buy", 1: "Buy", 0: "Hold", -1: "Sell", -2: "Strong Sell"}
SIGNAL_COLORS = {
    "Strong Buy":  "#00C853",
    "Buy":         "#69F0AE",
    "Hold":        "#90A4AE",
    "Sell":        "#FFAB91",
    "Strong Sell": "#FF1744",
}


# ═══════════════════════════════════════════════════════════════════
# LOADERS
# ═══════════════════════════════════════════════════════════════════
def _csv(path, **kw):
    try:
        if os.path.exists(path):
            return pd.read_csv(path, **kw)
    except Exception:
        pass
    return None

def _json(path):
    try:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    except Exception:
        pass
    return None

def discover_symbols():
    if not os.path.isdir(RESULTS_DIR):
        return []
    return sorted(
        n for n in os.listdir(RESULTS_DIR)
        if os.path.isdir(os.path.join(RESULTS_DIR, n))
        and os.path.exists(os.path.join(RESULTS_DIR, n, "predictions.csv"))
    )

def load_bundle(sym):
    b = os.path.join(RESULTS_DIR, sym)
    return dict(
        pred    = _csv(os.path.join(b, "predictions.csv"), parse_dates=["date"], index_col="date"),
        roll    = _csv(os.path.join(b, "rolling_error.csv"), parse_dates=[0], index_col=0),
        sig     = _csv(os.path.join(b, "signals.csv"), parse_dates=[0], index_col=0),
        t1      = _csv(os.path.join(b, "trades_mode1.csv")),
        t2      = _csv(os.path.join(b, "trades_mode2.csv")),
        summary = _json(os.path.join(b, "summary.json")),
    )

def leaderboard_df(overall):
    rows = []
    for sym, r in (overall or {}).items():
        if not isinstance(r, dict): continue
        m  = r.get("metrics") or {}
        m1 = r.get("mode1")   or {}
        m2 = r.get("mode2")   or {}
        rows.append(dict(
            Symbol       = sym,
            Status       = r.get("status", "?"),
            MAE          = m.get("MAE"),
            RMSE         = m.get("RMSE"),
            MAPE_pct     = m.get("MAPE"),
            R2           = m.get("R2"),
            Corr         = m.get("Correlation"),
            DirAcc_pct   = m.get("DirectionalAccuracy"),
            HitRatio_pct = m.get("HitRatio"),
            N            = m.get("N"),
            M1_NetPnL    = m1.get("net_pnl"),
            M1_Ret_pct   = m1.get("total_return_pct"),
            M1_Trades    = m1.get("n_trades"),
            M2_NetPnL    = m2.get("net_pnl"),
            M2_Ret_pct   = m2.get("total_return_pct"),
            M2_Trades    = m2.get("n_trades"),
        ))
    df = pd.DataFrame(rows)
    if not df.empty:
        for c in df.columns:
            if c not in ("Symbol", "Status"):
                df[c] = pd.to_numeric(df[c], errors="coerce").round(3)
    return df

def reconstruct_equity(pred, trades):
    if pred is None or pred.empty or "close_today" not in pred.columns:
        return pd.Series(dtype=float)
    px = pred["close_today"].sort_index()
    if trades is None or trades.empty:
        return pd.Series(CAPITAL, index=px.index)
    trades = trades.copy()
    trades["entry_date"] = pd.to_datetime(trades["entry_date"])
    trades["exit_date"]  = pd.to_datetime(trades["exit_date"])
    sorted_t = trades.sort_values("entry_date").to_dict("records")
    realized = CAPITAL; active = None; ti = 0; eq = []
    for date in px.index:
        while ti < len(sorted_t) and sorted_t[ti]["entry_date"] <= date and active is None:
            active = sorted_t[ti]; ti += 1
        if active is not None and active["exit_date"] == date:
            realized += float(active.get("net_pnl") or 0)
            active = None
        mtm = realized
        if active is not None:
            qty = active.get("qty", 0) or 0
            mtm = realized + qty * (float(px.loc[date]) - float(active["entry_price"]))
        eq.append(mtm)
    return pd.Series(eq, index=px.index)


# ═══════════════════════════════════════════════════════════════════
# FIGURE BUILDERS  (each returns a plotly Figure)
# ═══════════════════════════════════════════════════════════════════
def _base(fig, title, h=420):
    fig.update_layout(
        template=TPL, paper_bgcolor=PAPER, plot_bgcolor=BG,
        title=dict(text=title, font=dict(size=13, color=FONT)),
        font=dict(color=FONT, family="'JetBrains Mono','Fira Code',monospace"),
        height=h, margin=dict(l=50, r=30, t=50, b=40),
        legend=dict(orientation="h", y=1.08, bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=BORDER, showspikes=True,
                     spikecolor=SUB, spikethickness=1)
    fig.update_yaxes(gridcolor=GRID, linecolor=BORDER)
    return fig

def empty_fig(msg="No data", h=300):
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font=dict(size=14, color=SUB),
                       x=0.5, y=0.5, xref="paper", yref="paper")
    return _base(fig, "", h)

# ── 1. Leaderboard bars ──────────────────────────────────────────
def fig_leader_bars(df, col, title, ascending=False):
    if df.empty or col not in df.columns:
        return empty_fig(f"No data ({col})")
    d = df.dropna(subset=[col]).sort_values(col, ascending=ascending).head(25)
    if d.empty: return empty_fig()
    colors = [BULL if v >= 0 else BEAR for v in d[col]] \
        if any(k in col for k in ("Ret", "PnL")) else BLUE
    fig = go.Figure(go.Bar(x=d["Symbol"], y=d[col], marker_color=colors,
                           text=d[col].round(2), textposition="outside",
                           textfont=dict(size=9)))
    return _base(fig, title, 400)

# ── 2. 45-degree scatter ─────────────────────────────────────────
def fig_scatter_45(pred, sym):
    if pred is None or pred.empty:
        return empty_fig("No predictions.csv")
    df = pred.dropna(subset=["final_pred", "actual_next_close"])
    if df.empty: return empty_fig("No completed predictions")
    x = df["actual_next_close"].values
    y = df["final_pred"].values
    lo, hi = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
    ss_res = float(np.sum((y - x) ** 2))
    ss_tot = float(np.sum((x - x.mean()) ** 2))
    r2 = (1 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")

    # colour by date order (warm=recent)
    n = len(x)
    clr = list(range(n))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers", name="Prediction",
        marker=dict(size=5, color=clr, colorscale="Viridis", opacity=0.6,
                    colorbar=dict(title="Time →", thickness=10, len=0.6,
                                  tickfont=dict(size=9))),
        hovertemplate="Date: %{customdata}<br>Actual: ₹%{x:,.2f}<br>Pred: ₹%{y:,.2f}<extra></extra>",
        customdata=df.index.astype(str),
    ))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                             name="Perfect 45°",
                             line=dict(color=BEAR, width=2, dash="dash")))
    if len(x) > 1:
        b, a = np.polyfit(x, y, 1)
        fig.add_trace(go.Scatter(
            x=[lo, hi], y=[b*lo+a, b*hi+a], mode="lines",
            name=f"Best-fit  R²={r2:.4f}",
            line=dict(color=GOLD, width=2)))
    fig.update_xaxes(title="Actual next-day close (₹)")
    fig.update_yaxes(title="Predicted next-day close (₹)")
    return _base(fig, f"{sym} — Predicted vs Actual  (R²={r2:.4f})", 500)

# ── 3. Predicted vs actual time series ───────────────────────────
def fig_pred_series(pred, sym):
    if pred is None or pred.empty: return empty_fig()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pred.index, y=pred["close_today"],
                             name="Close (today)", line=dict(color=BLUE, width=1.3)))
    if "final_pred" in pred.columns:
        fig.add_trace(go.Scatter(x=pred.index, y=pred["final_pred"].shift(-1),
                                 name="Predicted (next-day)", opacity=0.8,
                                 line=dict(color=GOLD, width=1.1, dash="dot")))
    fig.update_yaxes(title="₹")
    return _base(fig, f"{sym} — Close vs Predicted Next-Day Close", 420)

# ── 4. Rolling error ─────────────────────────────────────────────
def fig_rolling(roll, sym):
    if roll is None or roll.empty: return empty_fig("No rolling_error.csv")
    fig = go.Figure()
    if "rolling_mae" in roll.columns:
        fig.add_trace(go.Scatter(x=roll.index, y=roll["rolling_mae"],
                                 name="Rolling MAE (20d)", line=dict(color=BLUE, width=1.5)))
    if "rolling_rmse" in roll.columns:
        fig.add_trace(go.Scatter(x=roll.index, y=roll["rolling_rmse"],
                                 name="Rolling RMSE (20d)",
                                 line=dict(color=ORANGE, width=1.5, dash="dot")))
    fig.update_yaxes(title="₹ error")
    return _base(fig, f"{sym} — Rolling Prediction Error", 380)

# ── 5. Signal distribution ───────────────────────────────────────
def fig_sig_dist(sig, sym):
    if sig is None or sig.empty or "signal" not in sig.columns:
        return empty_fig("No signals.csv")
    labels = sig["signal"].map(SIGNAL_LABELS).fillna("Hold")
    order = ["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"]
    counts = labels.value_counts().reindex(order).fillna(0)
    fig = go.Figure(go.Bar(
        x=counts.index, y=counts.values,
        marker_color=[SIGNAL_COLORS[k] for k in counts.index],
        text=counts.values.astype(int), textposition="outside",
    ))
    fig.update_yaxes(title="Days")
    return _base(fig, f"{sym} — Signal Distribution", 360)

# ── 6. Signal timeline ───────────────────────────────────────────
def fig_sig_timeline(sig, sym):
    if sig is None or sig.empty or "close" not in sig.columns:
        return empty_fig()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sig.index, y=sig["close"], name="Close",
                             line=dict(color=SUB, width=1.1)))
    for lab, code in [("Strong Buy",2),("Buy",1),("Sell",-1),("Strong Sell",-2)]:
        m = sig["signal"] == code
        if m.any():
            fig.add_trace(go.Scatter(
                x=sig.index[m], y=sig.loc[m, "close"], mode="markers",
                name=lab,
                marker=dict(size=7, color=SIGNAL_COLORS[lab],
                            symbol="triangle-up" if code > 0 else "triangle-down",
                            line=dict(width=0.8, color="white")),
                hovertemplate=f"{lab}<br>%{{x|%Y-%m-%d}}<br>₹%{{y:,.2f}}<extra></extra>",
            ))
    fig.update_yaxes(title="₹")
    return _base(fig, f"{sym} — Buy / Sell Signals on Price", 460)

# ── 7. Equity curve ──────────────────────────────────────────────
def fig_equity(eq, title, color=BULL):
    if eq is None or eq.empty: return empty_fig("No trade data")
    ret_pct = (eq.iloc[-1] / eq.iloc[0] - 1) * 100 if eq.iloc[0] != 0 else 0
    sign = "+" if ret_pct >= 0 else ""
    col = BULL if ret_pct >= 0 else BEAR
    fig = go.Figure(go.Scatter(x=eq.index, y=eq.values,
                               line=dict(color=color, width=1.6),
                               fill="tozeroy",
                               fillcolor=f"rgba({','.join(str(int(c,16)) for c in [color[1:3],color[3:5],color[5:7]])},0.07)",
                               hovertemplate="₹%{y:,.0f}<extra></extra>"))
    fig.add_annotation(text=f"Return: {sign}{ret_pct:.1f}%",
                       x=0.02, y=0.95, xref="paper", yref="paper",
                       showarrow=False, font=dict(size=13, color=col))
    fig.update_yaxes(title="₹")
    return _base(fig, title, 380)

# ── 8. Portfolio Mode 3 equity ────────────────────────────────────
def fig_portfolio_equity(t3):
    if t3 is None or t3.empty: return empty_fig("No trades_mode3_portfolio.csv")
    d = t3.copy()
    d["exit_date"] = pd.to_datetime(d["exit_date"], errors="coerce")
    d = d.sort_values("exit_date")
    cum = d["net_pnl"].fillna(0).cumsum() + CAPITAL
    ret = (cum.iloc[-1] / CAPITAL - 1) * 100
    sign = "+" if ret >= 0 else ""
    col = BULL if ret >= 0 else BEAR
    fig = go.Figure(go.Scatter(x=d["exit_date"], y=cum, line=dict(color=col, width=1.8),
                               fill="tozeroy",
                               fillcolor=f"rgba({','.join(str(int(c,16)) for c in [col[1:3],col[3:5],col[5:7]])},0.07)",
                               hovertemplate="%{x|%Y-%m-%d}<br>₹%{y:,.0f}<extra></extra>"))
    fig.add_annotation(text=f"Total: {sign}{ret:.1f}%  |  Final: ₹{cum.iloc[-1]:,.0f}",
                       x=0.02, y=0.95, xref="paper", yref="paper",
                       showarrow=False, font=dict(size=12, color=col))
    fig.update_yaxes(title="₹")
    return _base(fig, "Mode 3 — Portfolio Equity (cumulative realized P&L)", 440)

def fig_portfolio_contrib(t3):
    if t3 is None or t3.empty or "symbol" not in t3.columns: return empty_fig()
    by = t3.groupby("symbol")["net_pnl"].sum().sort_values(ascending=False).reset_index()
    colors = [BULL if v >= 0 else BEAR for v in by["net_pnl"]]
    fig = go.Figure(go.Bar(x=by["symbol"], y=by["net_pnl"], marker_color=colors,
                           text=by["net_pnl"].round(0), textposition="outside",
                           textfont=dict(size=9)))
    fig.update_xaxes(tickangle=-45)
    fig.update_yaxes(title="Net P&L (₹)")
    return _base(fig, "Mode 3 — Net P&L Contribution by Symbol", 400)

# ── 9. Prediction error residuals histogram ───────────────────────
def fig_residuals(pred, sym):
    if pred is None or pred.empty: return empty_fig()
    df = pred.dropna(subset=["final_pred", "actual_next_close"])
    if df.empty: return empty_fig()
    err = (df["final_pred"] - df["actual_next_close"]).values
    fig = go.Figure(go.Histogram(x=err, nbinsx=60,
                                 marker_color=BLUE, opacity=0.8,
                                 name="Residuals"))
    fig.add_vline(x=0, line_color=GOLD, line_dash="dash", line_width=1.5)
    fig.update_xaxes(title="Prediction Error (₹)")
    fig.update_yaxes(title="Count")
    return _base(fig, f"{sym} — Residuals Distribution (Pred − Actual)", 360)

# ── 10. Market regime histogram ───────────────────────────────────
def fig_regime_dist(sig, sym):
    if sig is None or sig.empty or "market_regime" not in sig.columns:
        return empty_fig("No market_regime column in signals.csv")
    rc = {"Crashing":"#EF5350","Falling":"#FF9800","Sideways":"#CE93D8",
          "Rising":"#42A5F5","Rally":"#00FF94"}
    vc = {"HighVol":"#EF5350","MedVol":"#FFD700","LowVol":"#00FF94"}
    order = ["Crashing","Falling","Sideways","Rising","Rally"]
    counts = sig["market_regime"].value_counts().reindex(order).fillna(0)
    fig = make_subplots(cols=2, subplot_titles=("Market Regime Days","Vol Regime Days"))
    fig.add_trace(go.Bar(x=counts.index, y=counts.values,
                         marker_color=[rc.get(k, SUB) for k in counts.index],
                         name="Market"), row=1, col=1)
    if "vol_regime" in sig.columns:
        vc_counts = sig["vol_regime"].value_counts().reindex(["HighVol","MedVol","LowVol"]).fillna(0)
        fig.add_trace(go.Bar(x=vc_counts.index, y=vc_counts.values,
                             marker_color=[vc.get(k, SUB) for k in vc_counts.index],
                             name="Vol"), row=1, col=2)
    fig.update_layout(template=TPL, paper_bgcolor=PAPER, plot_bgcolor=BG,
                      height=340, showlegend=False,
                      font=dict(color=FONT, family="'JetBrains Mono',monospace"),
                      title=dict(text=f"{sym} — Regime Breakdown", font=dict(size=13, color=FONT)),
                      margin=dict(l=50,r=30,t=60,b=40))
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID)
    return fig


# ═══════════════════════════════════════════════════════════════════
# HTML HELPERS
# ═══════════════════════════════════════════════════════════════════
def fig_to_div(fig, div_id=None):
    kwargs = dict(full_html=False, include_plotlyjs=False, config={"responsive": True})
    if div_id:
        kwargs["div_id"] = div_id
    return pio.to_html(fig, **kwargs)

def kpi_html(label, val, color=FONT, bg=CARD_BG):
    return f"""
<div class="kpi" style="background:{bg}">
  <div class="kpi-label">{label}</div>
  <div class="kpi-value" style="color:{color}">{val}</div>
</div>"""

def kpi_row(*kpis):
    return f'<div class="kpi-row">{"".join(kpis)}</div>'

def section(title, content, collapsed=False):
    state = "" if not collapsed else ' class="collapsed"'
    return f"""
<details open class="section"{state}>
  <summary class="section-title">▸ {title}</summary>
  <div class="section-body">{content}</div>
</details>"""

def grid2(*divs):
    inner = "".join(f'<div class="grid-cell">{d}</div>' for d in divs)
    return f'<div class="grid2">{inner}</div>'

def df_to_html_table(df, max_rows=200, fmt=None):
    if df is None or df.empty:
        return "<p class='no-data'>No data available.</p>"
    d = df.head(max_rows).copy()
    if fmt:
        for col, f in fmt.items():
            if col in d.columns:
                d[col] = d[col].apply(lambda x: f.format(x) if pd.notna(x) else "—")
    html = '<div class="tbl-wrap"><table class="data-tbl"><thead><tr>'
    for c in d.columns:
        html += f"<th>{c}</th>"
    html += "</tr></thead><tbody>"
    for _, row in d.iterrows():
        html += "<tr>"
        for c, v in row.items():
            cls = ""
            if isinstance(v, (int, float)) and pd.notna(v):
                if c in ("net_pnl","gross_pnl","M1_NetPnL","M2_NetPnL","M1_Ret_pct","M2_Ret_pct","total_return_pct"):
                    cls = ' class="pos"' if v >= 0 else ' class="neg"'
            html += f"<td{cls}>{v if pd.notna(v) else '—'}</td>"
        html += "</tr>"
    html += "</tbody></table></div>"
    if len(df) > max_rows:
        html += f"<p class='no-data'>Showing first {max_rows} of {len(df)} rows.</p>"
    return html

def metrics_table(m):
    if not m: return "<p class='no-data'>No metrics found.</p>"
    rows = ""
    labels = [("MAE","MAE (₹)"),("RMSE","RMSE (₹)"),("MAPE","MAPE (%)"),
              ("R2","R²"),("Correlation","Correlation"),
              ("DirectionalAccuracy","Directional Accuracy (%)"),
              ("HitRatio","Hit Ratio (%)"),("N","Observations")]
    for key, label in labels:
        v = m.get(key)
        if v is None: continue
        fmt = f"{v:.4f}" if isinstance(v, float) else str(v)
        rows += f"<tr><td class='metric-key'>{label}</td><td class='metric-val'>{fmt}</td></tr>"
    return f'<table class="metrics-tbl"><tbody>{rows}</tbody></table>'


# ═══════════════════════════════════════════════════════════════════
# MAIN  REPORT BUILDER
# ═══════════════════════════════════════════════════════════════════
def build_report():
    symbols = discover_symbols()
    overall = _json(os.path.join(RESULTS_DIR, "overall_summary.json")) or {}
    mode3   = _json(os.path.join(RESULTS_DIR, "mode3_summary.json"))
    t3      = _csv(os.path.join(RESULTS_DIR, "trades_mode3_portfolio.csv"))
    lb_df   = leaderboard_df(overall)
    now     = datetime.now().strftime("%Y-%m-%d %H:%M")

    print(f"  Found {len(symbols)} symbols: {symbols[:8]}{'...' if len(symbols)>8 else ''}")

    # ── Section 0: header KPIs ──────────────────────────────────
    n_ok  = int((lb_df["Status"] == "ok").sum()) if not lb_df.empty else 0
    n_err = int((lb_df["Status"] == "error").sum()) if not lb_df.empty else 0
    avg_r2    = lb_df["R2"].mean()   if not lb_df.empty else float("nan")
    avg_dir   = lb_df["DirAcc_pct"].mean() if not lb_df.empty else float("nan")
    avg_m1ret = lb_df["M1_Ret_pct"].mean() if not lb_df.empty else float("nan")
    avg_m2ret = lb_df["M2_Ret_pct"].mean() if not lb_df.empty else float("nan")

    hdr_kpis = kpi_row(
        kpi_html("Symbols Processed", str(len(symbols))),
        kpi_html("Successful", str(n_ok), color=BULL),
        kpi_html("Errors", str(n_err), color=BEAR if n_err else SUB),
        kpi_html("Avg R²", f"{avg_r2:.3f}" if pd.notna(avg_r2) else "—"),
        kpi_html("Avg Dir. Accuracy", f"{avg_dir:.1f}%" if pd.notna(avg_dir) else "—"),
        kpi_html("Avg Mode-1 Return", f"{avg_m1ret:+.1f}%" if pd.notna(avg_m1ret) else "—",
                 color=BULL if pd.notna(avg_m1ret) and avg_m1ret>=0 else BEAR),
        kpi_html("Avg Mode-2 Return", f"{avg_m2ret:+.1f}%" if pd.notna(avg_m2ret) else "—",
                 color=BULL if pd.notna(avg_m2ret) and avg_m2ret>=0 else BEAR),
    )

    # ── Section 1: leaderboard table + bar charts ────────────────
    lb_cols = ["Symbol","Status","MAE","RMSE","R2","DirAcc_pct","HitRatio_pct",
               "M1_NetPnL","M1_Ret_pct","M1_Trades","M2_NetPnL","M2_Ret_pct","M2_Trades"]
    lb_fmt  = {"M1_NetPnL":"₹{:,.0f}","M2_NetPnL":"₹{:,.0f}",
               "M1_Ret_pct":"{:+.2f}%","M2_Ret_pct":"{:+.2f}%"}
    lb_html = df_to_html_table(lb_df[lb_cols] if not lb_df.empty else lb_df, fmt=lb_fmt)

    lb_bar1 = fig_to_div(fig_leader_bars(lb_df, "M1_Ret_pct", "Mode 1 — Total Return % by Symbol"))
    lb_bar2 = fig_to_div(fig_leader_bars(lb_df, "M2_Ret_pct", "Mode 2 — Total Return % by Symbol"))
    lb_bar3 = fig_to_div(fig_leader_bars(lb_df, "DirAcc_pct", "Directional Accuracy % by Symbol"))
    lb_bar4 = fig_to_div(fig_leader_bars(lb_df, "R2",         "R² by Symbol"))
    lb_bar5 = fig_to_div(fig_leader_bars(lb_df, "MAE",        "MAE (₹) by Symbol", ascending=True))
    lb_bar6 = fig_to_div(fig_leader_bars(lb_df, "RMSE",       "RMSE (₹) by Symbol", ascending=True))

    sec_overview = (
        hdr_kpis +
        section("Cross-Symbol Leaderboard",
            lb_html + grid2(lb_bar1, lb_bar2) + grid2(lb_bar3, lb_bar4) + grid2(lb_bar5, lb_bar6))
    )

    # ── Section 2: per-symbol detail ────────────────────────────
    sym_sections = ""
    for i, sym in enumerate(symbols):
        pct = int((i+1)/len(symbols)*100)
        print(f"\r  Building symbol sections... {pct:3d}%  [{sym:<14s}]", end="", flush=True)

        bundle = load_bundle(sym)
        pred, roll, sig, t1, t2, summary = (
            bundle["pred"], bundle["roll"], bundle["sig"],
            bundle["t1"],   bundle["t2"],   bundle["summary"])

        m  = (summary or {}).get("metrics", {})
        m1 = (summary or {}).get("mode1",   {})
        m2 = (summary or {}).get("mode2",   {})

        # KPIs
        sym_kpis = kpi_row(
            kpi_html("MAE", f"₹{m.get('MAE',0):.3f}" if m.get("MAE") is not None else "—"),
            kpi_html("RMSE", f"₹{m.get('RMSE',0):.3f}" if m.get("RMSE") is not None else "—"),
            kpi_html("R²",   f"{m.get('R2',0):.4f}" if m.get("R2") is not None else "—"),
            kpi_html("Dir. Acc", f"{m.get('DirectionalAccuracy',0):.1f}%" if m.get("DirectionalAccuracy") is not None else "—"),
            kpi_html("Hit Ratio", f"{m.get('HitRatio',0):.1f}%" if m.get("HitRatio") is not None else "—"),
            kpi_html("Mode1 Net P&L",
                     f"₹{m1.get('net_pnl',0):,.0f}" if m1.get("net_pnl") is not None else "—",
                     color=BULL if (m1.get("net_pnl") or 0)>=0 else BEAR),
            kpi_html("Mode2 Net P&L",
                     f"₹{m2.get('net_pnl',0):,.0f}" if m2.get("net_pnl") is not None else "—",
                     color=BULL if (m2.get("net_pnl") or 0)>=0 else BEAR),
        )

        # Figures
        f_scatter  = fig_to_div(fig_scatter_45(pred, sym))
        f_resid    = fig_to_div(fig_residuals(pred, sym))
        f_series   = fig_to_div(fig_pred_series(pred, sym))
        f_roll     = fig_to_div(fig_rolling(roll, sym))
        f_sig_d    = fig_to_div(fig_sig_dist(sig, sym))
        f_sig_t    = fig_to_div(fig_sig_timeline(sig, sym))
        f_regime   = fig_to_div(fig_regime_dist(sig, sym))

        eq1 = reconstruct_equity(pred, t1)
        eq2 = reconstruct_equity(pred, t2)
        f_eq1 = fig_to_div(fig_equity(eq1, f"{sym} — Mode 1 Equity (Fixed 100-share Qty)", BULL))
        f_eq2 = fig_to_div(fig_equity(eq2, f"{sym} — Mode 2 Equity (Capital Compounding)", BLUE))

        t1_html = df_to_html_table(t1.round(2) if t1 is not None else None,
                                   fmt={"net_pnl":"₹{:,.2f}","gross_pnl":"₹{:,.2f}"})
        t2_html = df_to_html_table(t2.round(2) if t2 is not None else None,
                                   fmt={"net_pnl":"₹{:,.2f}","gross_pnl":"₹{:,.2f}"})

        sym_inner = (
            sym_kpis
            + section("Prediction Accuracy — 45° Scatter & Residuals",
                       grid2(f_scatter, f_resid))
            + section("Prediction Time Series & Rolling Error",
                       f_series + f_roll)
            + section("Signal Analysis",
                       grid2(f_sig_d, f_regime) + f_sig_t)
            + section("Equity Curves",
                       grid2(f_eq1, f_eq2))
            + section("Trade Log — Mode 1 (Fixed Qty 100 shares)",
                       t1_html, collapsed=True)
            + section("Trade Log — Mode 2 (Capital Compounding)",
                       t2_html, collapsed=True)
        )
        sym_sections += f'<div class="sym-block" id="sym-{sym}">'
        sym_sections += f'<h2 class="sym-title">◈ {sym}</h2>'
        sym_sections += sym_inner
        sym_sections += '</div>'

    print()  # newline after progress

    # ── Section 3: Portfolio (Mode 3) ────────────────────────────
    if mode3:
        p3_kpis = kpi_row(
            kpi_html("Net P&L",
                     f"₹{mode3.get('net_pnl',0):,.0f}",
                     color=BULL if (mode3.get("net_pnl") or 0)>=0 else BEAR),
            kpi_html("Gross P&L", f"₹{mode3.get('gross_pnl',0):,.0f}"),
            kpi_html("Total Return %",
                     f"{mode3.get('total_return_pct',0):+.2f}%",
                     color=BULL if (mode3.get("total_return_pct") or 0)>=0 else BEAR),
            kpi_html("Final Equity", f"₹{mode3.get('final_equity',0):,.0f}"),
            kpi_html("Total Trades", str(mode3.get("n_trades","—"))),
        )
        f_p3eq   = fig_to_div(fig_portfolio_equity(t3))
        f_p3cont = fig_to_div(fig_portfolio_contrib(t3))
        t3_html  = df_to_html_table(t3.round(2) if t3 is not None else None,
                                    fmt={"net_pnl":"₹{:,.2f}","gross_pnl":"₹{:,.2f}"})
        sec_portfolio = (p3_kpis
            + grid2(f_p3eq, f_p3cont)
            + section("Portfolio Trades (Mode 3)", t3_html, collapsed=True))
    else:
        sec_portfolio = "<p class='no-data'>No mode3_summary.json found.</p>"

    # ── Symbol nav sidebar items ─────────────────────────────────
    nav_items = "".join(
        f'<a href="#sym-{s}" class="nav-item">{s}</a>' for s in symbols)

    # ── Assemble full HTML ───────────────────────────────────────
    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>QBEAST SVR — Results Report  |  {now}</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js" charset="utf-8"></script>
<style>
/* ── reset ── */
*{{box-sizing:border-box;margin:0;padding:0}}
html{{scroll-behavior:smooth}}
body{{background:{BG};color:{FONT};
     font-family:'JetBrains Mono','Fira Code','Courier New',monospace;
     font-size:13px;line-height:1.55;display:flex;min-height:100vh}}

/* ── sidebar ── */
#sidebar{{width:160px;min-width:140px;background:#0a0f16;
          border-right:1px solid {BORDER};padding:12px 0;
          position:fixed;top:0;left:0;bottom:0;overflow-y:auto;z-index:100}}
#sidebar h3{{color:{ORANGE};font-size:11px;padding:10px 12px 6px;letter-spacing:1px}}
.nav-item{{display:block;padding:5px 12px;color:{SUB};text-decoration:none;
           font-size:11px;border-left:2px solid transparent;transition:all .15s}}
.nav-item:hover{{color:{FONT};border-color:{BLUE};background:{CARD_BG}}}
.nav-tabs{{display:flex;flex-direction:column;gap:2px;margin-top:6px}}
.nav-tab{{display:block;padding:7px 12px;cursor:pointer;font-size:11px;
          color:{SUB};border:none;background:none;text-align:left;
          border-left:2px solid transparent;transition:all .15s;width:100%}}
.nav-tab:hover,.nav-tab.active{{color:{FONT};border-color:{ORANGE};background:{CARD_BG}}}
.nav-sep{{border:none;border-top:1px solid {BORDER};margin:8px 12px}}

/* ── main ── */
#main{{margin-left:160px;padding:20px 24px;flex:1;max-width:calc(100vw - 160px)}}

/* ── page sections ── */
.page{{display:none}}.page.active{{display:block}}

/* ── header ── */
.report-hdr{{background:#0a0f16;border:1px solid {BORDER};border-radius:8px;
             padding:14px 18px;margin-bottom:18px}}
.report-hdr h1{{color:{ORANGE};font-size:17px;font-weight:bold;margin-bottom:4px}}
.report-meta{{color:{SUB};font-size:11px}}

/* ── KPIs ── */
.kpi-row{{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:14px}}
.kpi{{background:{CARD_BG};border:1px solid {BORDER};border-radius:8px;
      padding:10px 16px;min-width:110px}}
.kpi-label{{color:{SUB};font-size:9px;text-transform:uppercase;letter-spacing:.5px;margin-bottom:3px}}
.kpi-value{{font-size:17px;font-weight:bold}}

/* ── sections ── */
.section{{background:{CARD_BG};border:1px solid {BORDER};border-radius:8px;
          margin-bottom:12px;overflow:hidden}}
.section-title{{background:#0a0f16;padding:9px 14px;cursor:pointer;
                font-size:12px;color:{BLUE};letter-spacing:.3px;
                list-style:none;user-select:none}}
.section-title::-webkit-details-marker{{display:none}}
.section-title:hover{{color:{FONT}}}
.section-body{{padding:12px}}
details[open]>.section-title::before{{content:"▾ "}}
details:not([open])>.section-title::before{{content:"▸ "}}
details>.section-title{{padding-left:10px}}

/* ── sym blocks ── */
.sym-block{{margin-bottom:24px}}
.sym-title{{color:{ORANGE};font-size:15px;font-weight:bold;
            margin-bottom:10px;padding-bottom:6px;
            border-bottom:1px solid {BORDER}}}

/* ── grids ── */
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:10px}}
.grid-cell{{min-width:0}}

/* ── tables ── */
.tbl-wrap{{overflow-x:auto;margin-bottom:6px}}
.data-tbl{{width:100%;border-collapse:collapse;font-size:10.5px}}
.data-tbl th{{background:#0a0f16;color:{SUB};border:1px solid {BORDER};
              padding:5px 8px;text-align:left;font-weight:normal;
              white-space:nowrap}}
.data-tbl td{{border:1px solid {BORDER};padding:4px 8px;
              white-space:nowrap;color:{FONT}}}
.data-tbl tr:nth-child(even) td{{background:rgba(255,255,255,.02)}}
.data-tbl tr:hover td{{background:rgba(66,165,245,.05)}}
td.pos{{color:{BULL}}}
td.neg{{color:{BEAR}}}

/* ── metrics mini table ── */
.metrics-tbl{{border-collapse:collapse;min-width:340px}}
.metrics-tbl td{{padding:5px 12px;border-bottom:1px solid {BORDER};font-size:12px}}
.metric-key{{color:{SUB};width:200px}}
.metric-val{{color:{FONT};font-weight:bold}}

/* ── misc ── */
.no-data{{color:{SUB};font-size:11px;padding:10px 0;font-style:italic}}
::-webkit-scrollbar{{width:6px;height:6px}}
::-webkit-scrollbar-thumb{{background:{BORDER};border-radius:3px}}

/* ── responsive ── */
@media(max-width:900px){{
  .grid2{{grid-template-columns:1fr}}
  #sidebar{{display:none}}
  #main{{margin-left:0}}
}}
</style>
</head>
<body>

<!-- ── SIDEBAR ─────────────────────────────────────────── -->
<nav id="sidebar">
  <h3>QBEAST SVR</h3>
  <div class="nav-tabs">
    <button class="nav-tab active" onclick="showPage('overview')">📊 Overview</button>
    <button class="nav-tab" onclick="showPage('portfolio')">🗂 Portfolio (M3)</button>
    <button class="nav-tab" onclick="showPage('symbols')">📈 Symbols</button>
  </div>
  <hr class="nav-sep">
  <div style="padding:4px 12px;color:{SUB};font-size:9px;text-transform:uppercase;
              letter-spacing:.5px;margin-bottom:4px">Jump to symbol</div>
  {nav_items}
</nav>

<!-- ── MAIN ─────────────────────────────────────────────── -->
<main id="main">

<div class="report-hdr">
  <h1>QBEAST AI — SVR Walk-Forward Backtest Report</h1>
  <div class="report-meta">
    Generated: {now} &nbsp;|&nbsp; Results: {RESULTS_DIR} &nbsp;|&nbsp;
    Symbols: {len(symbols)} &nbsp;|&nbsp; Capital: ₹{CAPITAL:,.0f} &nbsp;|&nbsp;
    Backtest: Jan 2019 → Jun 2026
  </div>
</div>

<!-- PAGE: OVERVIEW -->
<div id="page-overview" class="page active">
  <h2 class="sym-title">Overview &amp; Cross-Symbol Leaderboard</h2>
  {sec_overview}
</div>

<!-- PAGE: PORTFOLIO -->
<div id="page-portfolio" class="page">
  <h2 class="sym-title">Portfolio-Level Backtest (Mode 3)</h2>
  {sec_portfolio}
</div>

<!-- PAGE: SYMBOLS -->
<div id="page-symbols" class="page">
  {sym_sections}
</div>

</main>

<script>
function showPage(name) {{
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
  document.getElementById('page-' + name).classList.add('active');
  event.target.classList.add('active');
  // resize all plotly charts so they don't remain squished
  setTimeout(() => {{
    document.querySelectorAll('.plotly-graph-div').forEach(d => {{
      if (window.Plotly) Plotly.Plots.resize(d);
    }});
  }}, 100);
}}

// Nav item clicks go to symbol page automatically
document.querySelectorAll('.nav-item').forEach(a => {{
  a.addEventListener('click', () => {{
    // activate symbol page
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
    document.getElementById('page-symbols').classList.add('active');
    document.querySelectorAll('.nav-tab')[2].classList.add('active');
    setTimeout(() => {{
      document.querySelectorAll('.plotly-graph-div').forEach(d => {{
        if (window.Plotly) Plotly.Plots.resize(d);
      }});
    }}, 150);
  }});
}});
</script>
</body>
</html>"""

    return html_doc


# ═══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  QBEAST SVR — HTML Report Generator")
    print(f"  Results dir : {RESULTS_DIR}")
    print(f"  Output file : {OUT_PATH}")
    print(f"{'='*60}")

    if not os.path.isdir(RESULTS_DIR):
        print(f"\n  ERROR: results dir not found: {RESULTS_DIR}")
        print("  Run run_pipeline.py first, or set --results-dir correctly.")
        sys.exit(1)

    html = build_report()

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)

    size_kb = os.path.getsize(OUT_PATH) / 1024
    print(f"\n  ✓ Report saved → {os.path.abspath(OUT_PATH)}")
    print(f"    File size : {size_kb:.0f} KB")
    print(f"    Open it in any browser — no server needed.\n")