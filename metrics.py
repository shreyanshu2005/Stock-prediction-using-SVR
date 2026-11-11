"""
metrics.py
=====================================================================
Prediction-quality evaluation for the walk-forward SVR: MAE, RMSE,
MAPE, R^2, Directional Accuracy, Hit Ratio, Correlation, rolling
MAE/RMSE, and a predicted-vs-actual scatter plot (matplotlib, saved
to disk -- no display dependency required).
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def compute_metrics(pred_df: pd.DataFrame, pred_col: str = "final_pred",
                    actual_col: str = "actual_next_close",
                    close_col: str = "close_today") -> dict:
    df = pred_df.dropna(subset=[pred_col, actual_col]).copy()
    if df.empty:
        return {}
    y_true = df[actual_col].values
    y_pred = df[pred_col].values

    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mape = float(np.mean(np.abs(err / y_true)) * 100.0)
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(df) > 2 else np.nan

    # directional accuracy: did predicted direction (vs today's close) match
    # actual direction (vs today's close)?
    if close_col in df.columns:
        today = df[close_col].values
        pred_dir = np.sign(y_pred - today)
        true_dir = np.sign(y_true - today)
        dir_acc = float(np.mean(pred_dir == true_dir)) * 100.0
        hit_ratio = float(np.mean((pred_dir != 0) & (pred_dir == true_dir) & (true_dir != 0))) * 100.0
    else:
        dir_acc = hit_ratio = np.nan

    return dict(MAE=mae, RMSE=rmse, MAPE=mape, R2=r2, Correlation=corr,
               DirectionalAccuracy=dir_acc, HitRatio=hit_ratio, N=len(df))


def rolling_error(pred_df: pd.DataFrame, window: int = 20,
                  pred_col: str = "final_pred",
                  actual_col: str = "actual_next_close") -> pd.DataFrame:
    df = pred_df.copy()
    err = (df[pred_col] - df[actual_col])
    out = pd.DataFrame(index=df.index)
    out["rolling_mae"] = err.abs().rolling(window).mean()
    out["rolling_rmse"] = (err ** 2).rolling(window).mean().pow(0.5)
    return out


def plot_scatter(pred_df: pd.DataFrame, out_path: str, title: str = "",
                 pred_col: str = "final_pred", actual_col: str = "actual_next_close"):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---- dark theme palette (matches the QBEAST dashboard scatter style) ----
    BG = "#15171c"
    GRID = "#2a2e38"
    TEXT = "#cfd3dc"
    SCATTER_COLOR = "#16c79a"   # teal-green dots
    SCATTER_EDGE = "#0c8f6e"
    PERFECT_COLOR = "#ff3b3b"   # red dashed "Perfect 45°"
    FIT_COLOR = "#ffc400"       # gold "R²=..." fit line

    df = pred_df.dropna(subset=[pred_col, actual_col])
    x = df[actual_col].values.astype(float)
    y = df[pred_col].values.astype(float)

    # R^2 of predictions vs actuals (1 - SS_res/SS_tot), same definition as compute_metrics
    ss_res = float(np.sum((y - x) ** 2))
    ss_tot = float(np.sum((x - x.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")

    fig, ax = plt.subplots(figsize=(6, 6.2))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())

    # scatter
    ax.scatter(x, y, s=10, alpha=0.55, color=SCATTER_COLOR,
              edgecolors=SCATTER_EDGE, linewidths=0.3, zorder=2)

    # perfect 45-degree reference line (dashed red)
    ax.plot([lo, hi], [lo, hi], color=PERFECT_COLOR, lw=1.6, ls="--",
           marker="o", markersize=2, markevery=0.08, label="Perfect 45°", zorder=3)

    # best-fit line through the data (gold), labeled with R^2
    if len(x) > 1:
        b, a = np.polyfit(x, y, 1)   # y = b*x + a
        fit_x = np.array([lo, hi])
        ax.plot(fit_x, b * fit_x + a, color=FIT_COLOR, lw=2.0,
               label=f"R²={r2:.3f}", zorder=4)
    else:
        ax.plot([], [], color=FIT_COLOR, lw=2.0, label=f"R²={r2:.3f}")

    # styling to match dark dashboard look
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.grid(color=GRID, alpha=0.5, lw=0.6)
    ax.set_axisbelow(True)

    ax.set_xlabel("Actual next-day close", color=TEXT, fontsize=10)
    ax.set_ylabel("Predicted next-day close", color=TEXT, fontsize=10)
    ax.set_title(title or "Predicted vs Actual", color="#5ad1ff", fontsize=12, pad=10)

    legend = ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=TEXT, loc="upper left",
                       fontsize=9)
    legend.get_frame().set_alpha(0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, facecolor=BG)
    plt.close(fig)
    return out_path


def plot_equity_curve(equity: pd.Series, out_path: str, title: str = "Equity Curve"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(equity.index, equity.values, color="#00C853", lw=1.4)
    ax.set_title(title)
    ax.set_ylabel("Equity (₹)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path