"""
charts.py
---------
Generates performance charts comparing the portfolio vs SPY benchmark.

Charts produced:
  1. NAV growth — portfolio vs SPY over time
  2. Drawdown — underwater chart for both
  3. Rolling Sharpe — 12-month rolling Sharpe ratio
  4. Monthly returns heatmap — year x month grid
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import TwoSlopeNorm
import os

from src.backtest.metrics import (
    compute_drawdown_series,
    compute_monthly_returns,
)


def plot_nav(
    portfolio_nav: pd.Series,
    benchmark_nav: pd.Series,
    save_path: str = "data/charts/nav_comparison.png",
) -> None:
    """Plot NAV growth of portfolio vs SPY."""
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(portfolio_nav.index, portfolio_nav.values,
            label="MVO Portfolio", color="#1f77b4", linewidth=1.5)
    ax.plot(benchmark_nav.index, benchmark_nav.values,
            label="SPY (Buy & Hold)", color="#ff7f0e",
            linewidth=1.5, linestyle="--")

    ax.set_title("Portfolio NAV vs SPY", fontsize=14, fontweight="bold")
    ax.set_ylabel("NAV ($)")
    ax.set_xlabel("")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Chart saved: {save_path}")


def plot_drawdown(
    portfolio_nav: pd.Series,
    benchmark_nav: pd.Series,
    save_path: str = "data/charts/drawdown.png",
) -> None:
    """Plot drawdown (underwater) chart for portfolio vs SPY."""
    port_dd = compute_drawdown_series(portfolio_nav) * 100
    bench_dd = compute_drawdown_series(benchmark_nav) * 100

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.fill_between(port_dd.index, port_dd.values, 0,
                    alpha=0.4, color="#1f77b4", label="MVO Portfolio")
    ax.fill_between(bench_dd.index, bench_dd.values, 0,
                    alpha=0.3, color="#ff7f0e", label="SPY")
    ax.plot(port_dd.index, port_dd.values, color="#1f77b4", linewidth=0.8)
    ax.plot(bench_dd.index, bench_dd.values, color="#ff7f0e",
            linewidth=0.8, linestyle="--")

    ax.set_title("Drawdown", fontsize=14, fontweight="bold")
    ax.set_ylabel("Drawdown (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Chart saved: {save_path}")


def plot_rolling_sharpe(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = 252,
    risk_free_rate: float = 0.05,
    save_path: str = "data/charts/rolling_sharpe.png",
) -> None:
    """Plot 12-month rolling Sharpe ratio for portfolio vs SPY."""
    daily_rf = risk_free_rate / 252

    port_excess = portfolio_returns - daily_rf
    bench_excess = benchmark_returns - daily_rf

    port_sharpe = (
        port_excess.rolling(window).mean() /
        port_excess.rolling(window).std()
    ) * np.sqrt(252)

    bench_sharpe = (
        bench_excess.rolling(window).mean() /
        bench_excess.rolling(window).std()
    ) * np.sqrt(252)

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(port_sharpe.index, port_sharpe.values,
            label="MVO Portfolio", color="#1f77b4", linewidth=1.5)
    ax.plot(bench_sharpe.index, bench_sharpe.values,
            label="SPY", color="#ff7f0e",
            linewidth=1.5, linestyle="--")
    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="-")
    ax.axhline(y=1, color="green", linewidth=0.8, linestyle=":",
               alpha=0.5, label="Sharpe = 1")

    ax.set_title("Rolling 12-Month Sharpe Ratio", fontsize=14, fontweight="bold")
    ax.set_ylabel("Sharpe Ratio")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Chart saved: {save_path}")


def plot_monthly_returns_heatmap(
    portfolio_returns: pd.Series,
    save_path: str = "data/charts/monthly_heatmap.png",
) -> None:
    """Plot monthly returns heatmap — year x month grid."""
    monthly = compute_monthly_returns(portfolio_returns) * 100
    monthly_df = monthly.to_frame("return")
    monthly_df["year"] = monthly_df.index.year
    monthly_df["month"] = monthly_df.index.month

    pivot = monthly_df.pivot(index="year", columns="month", values="return")
    pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig, ax = plt.subplots(figsize=(14, len(pivot) * 0.7 + 1))
    norm = TwoSlopeNorm(vmin=pivot.min().min(), vcenter=0, vmax=pivot.max().max())
    im = ax.imshow(pivot.values, cmap="RdYlGn", norm=norm, aspect="auto")

    ax.set_xticks(range(12))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index)

    # Annotate cells with return values
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}%",
                        ha="center", va="center",
                        fontsize=8,
                        color="black" if abs(val) < 8 else "white")

    plt.colorbar(im, ax=ax, label="Monthly Return (%)")
    ax.set_title("Portfolio Monthly Returns Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Chart saved: {save_path}")


def plot_all(
    portfolio_nav: pd.Series,
    benchmark_nav: pd.Series,
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.05,
    charts_dir: str = "data/charts",
) -> None:
    """Generate all four charts."""
    print("\nGenerating charts...")
    plot_nav(portfolio_nav, benchmark_nav,
             save_path=f"{charts_dir}/nav_comparison.png")
    plot_drawdown(portfolio_nav, benchmark_nav,
                  save_path=f"{charts_dir}/drawdown.png")
    plot_rolling_sharpe(portfolio_returns, benchmark_returns,
                        risk_free_rate=risk_free_rate,
                        save_path=f"{charts_dir}/rolling_sharpe.png")
    plot_monthly_returns_heatmap(portfolio_returns,
                                  save_path=f"{charts_dir}/monthly_heatmap.png")
    print(f"\nAll charts saved to {charts_dir}/")
