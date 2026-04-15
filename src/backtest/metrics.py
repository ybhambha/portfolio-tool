"""
metrics.py
----------
Computes performance statistics for the portfolio and benchmark.

Metrics computed:
  - Annualized return (CAGR)
  - Annualized volatility
  - Sharpe ratio
  - Maximum drawdown
  - Average drawdown
  - Calmar ratio (return / max drawdown)
  - Win rate (% of months portfolio beat benchmark)
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def compute_cagr(nav: pd.Series) -> float:
    """
    Compound Annual Growth Rate.
    CAGR = (Final NAV / Initial NAV) ^ (1 / years) - 1
    """
    n_days = len(nav)
    n_years = n_days / 252
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / n_years) - 1
    return cagr


def compute_volatility(daily_returns: pd.Series) -> float:
    """Annualized volatility = std(daily returns) * sqrt(252)"""
    return daily_returns.std() * np.sqrt(252)


def compute_sharpe(daily_returns: pd.Series, risk_free_rate: float = 0.05) -> float:
    """
    Sharpe ratio = (annualized return - risk free rate) / annualized volatility
    Uses daily returns to compute both numerator and denominator.
    """
    ann_return = daily_returns.mean() * 252
    ann_vol = daily_returns.std() * np.sqrt(252)
    if ann_vol == 0:
        return 0.0
    return (ann_return - risk_free_rate) / ann_vol


def compute_drawdown_series(nav: pd.Series) -> pd.Series:
    """
    Compute the drawdown series.
    Drawdown = (NAV - rolling maximum NAV) / rolling maximum NAV
    Always <= 0.
    """
    rolling_max = nav.cummax()
    drawdown = (nav - rolling_max) / rolling_max
    return drawdown


def compute_max_drawdown(nav: pd.Series) -> float:
    """Maximum drawdown — worst peak to trough decline."""
    drawdown = compute_drawdown_series(nav)
    return drawdown.min()


def compute_avg_drawdown(nav: pd.Series) -> float:
    """Average drawdown — mean of all drawdown values below zero."""
    drawdown = compute_drawdown_series(nav)
    negative_dd = drawdown[drawdown < 0]
    if len(negative_dd) == 0:
        return 0.0
    return negative_dd.mean()


def compute_calmar(nav: pd.Series) -> float:
    """
    Calmar ratio = CAGR / abs(Max Drawdown)
    Higher is better. Measures return per unit of drawdown risk.
    """
    cagr = compute_cagr(nav)
    max_dd = abs(compute_max_drawdown(nav))
    if max_dd == 0:
        return 0.0
    return cagr / max_dd


def compute_all_metrics(
    portfolio_nav: pd.Series,
    portfolio_returns: pd.Series,
    benchmark_nav: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.05,
) -> pd.DataFrame:
    """
    Compute all performance metrics for portfolio and benchmark side by side.

    Returns
    -------
    pd.DataFrame : metrics comparison table
    """
    metrics = {}

    for label, nav, returns in [
        ("Portfolio", portfolio_nav, portfolio_returns),
        ("SPY", benchmark_nav, benchmark_returns),
    ]:
        cagr = compute_cagr(nav)
        vol = compute_volatility(returns)
        sharpe = compute_sharpe(returns, risk_free_rate)
        max_dd = compute_max_drawdown(nav)
        avg_dd = compute_avg_drawdown(nav)
        calmar = compute_calmar(nav)

        metrics[label] = {
            "Annualized Return (%)": round(cagr * 100, 2),
            "Annualized Volatility (%)": round(vol * 100, 2),
            "Sharpe Ratio": round(sharpe, 3),
            "Max Drawdown (%)": round(max_dd * 100, 2),
            "Avg Drawdown (%)": round(avg_dd * 100, 2),
            "Calmar Ratio": round(calmar, 3),
        }

    df = pd.DataFrame(metrics)
    df["Difference"] = df["Portfolio"] - df["SPY"]
    return df


def compute_monthly_returns(daily_returns: pd.Series) -> pd.Series:
    """Convert daily returns to monthly returns."""
    return (1 + daily_returns).resample("ME").prod() - 1


def compute_win_rate(
    portfolio_daily: pd.Series,
    benchmark_daily: pd.Series,
) -> float:
    """
    % of months where portfolio outperformed benchmark.
    """
    port_monthly = compute_monthly_returns(portfolio_daily)
    bench_monthly = compute_monthly_returns(benchmark_daily)

    # Align on common dates
    common = port_monthly.index.intersection(bench_monthly.index)
    wins = (port_monthly[common] > bench_monthly[common]).sum()
    return wins / len(common) * 100


def print_metrics_table(metrics_df: pd.DataFrame, win_rate: float) -> None:
    """Pretty print the metrics comparison table."""
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    print(metrics_df.to_string())
    print(f"\nMonthly Win Rate vs SPY: {win_rate:.1f}%")
    print("=" * 60 + "\n")
