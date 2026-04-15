"""
simulator.py
------------
Simulates portfolio NAV and benchmark NAV over the backtest period.

At each rebalance date:
  1. Apply new target weights
  2. Deduct transaction costs on turnover
  3. Let the portfolio drift with market returns until next rebalance

Between rebalance dates:
  - Portfolio drifts naturally with daily returns
  - No trading occurs (buy and hold between rebalances)
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def simulate_portfolio(
    simple_returns: pd.DataFrame,
    weights_df: pd.DataFrame,
    etfs: list[str],
    transaction_cost_bps: float = 5.0,
    initial_nav: float = 10_000.0,
) -> pd.DataFrame:
    """
    Simulate portfolio NAV using walk-forward weights.

    Parameters
    ----------
    simple_returns : pd.DataFrame
        Daily simple returns for all assets, shape (days, tickers)
    weights_df : pd.DataFrame
        Target weights at each rebalance date, shape (n_rebalances, n_etfs)
    etfs : list[str]
        List of ETF tickers in the portfolio
    transaction_cost_bps : float
        Transaction cost per trade in basis points
    initial_nav : float
        Starting portfolio value (default $10,000)

    Returns
    -------
    pd.DataFrame with columns:
        nav          : portfolio NAV each day
        daily_return : daily portfolio return
        weights      : current weights (as dict per row)
    """
    cost_per_unit = transaction_cost_bps / 10_000

    # Only use ETF returns
    etf_returns = simple_returns[etfs]

    # Get all trading days in the backtest period
    # Start from the first rebalance date
    first_rebal = weights_df.index[0]
    trading_days = etf_returns.loc[first_rebal:].index

    # Initialize
    nav = initial_nav
    current_weights = weights_df.iloc[0].values  # equal-ish from optimizer
    rebalance_dates = set(weights_df.index)

    records = []

    for i, date in enumerate(trading_days):
        # --- Rebalance if this is a rebalance date ---
        if date in rebalance_dates and i > 0:
            new_weights = weights_df.loc[date].values

            # Transaction cost = cost_per_unit * sum(|new - old|) * NAV
            turnover = np.abs(new_weights - current_weights).sum()
            tc = cost_per_unit * turnover * nav
            nav -= tc  # deduct transaction cost from NAV

            current_weights = new_weights

        # --- Apply daily returns ---
        day_returns = etf_returns.loc[date].values
        portfolio_return = float(current_weights @ day_returns)
        nav = nav * (1 + portfolio_return)

        # --- Update weights for drift ---
        # Weights drift naturally with returns between rebalances
        new_w = current_weights * (1 + day_returns)
        current_weights = new_w / new_w.sum()  # renormalize

        records.append({
            "date": date,
            "nav": nav,
            "daily_return": portfolio_return,
        })

    result = pd.DataFrame(records).set_index("date")
    logger.info(
        f"Portfolio simulation complete | "
        f"{len(result)} days | "
        f"Final NAV: ${result['nav'].iloc[-1]:,.2f}"
    )
    return result


def simulate_benchmark(
    simple_returns: pd.DataFrame,
    benchmark: str,
    start_date: pd.Timestamp,
    initial_nav: float = 10_000.0,
) -> pd.DataFrame:
    """
    Simulate buy-and-hold benchmark (SPY) NAV.

    Parameters
    ----------
    simple_returns : pd.DataFrame
        Daily simple returns for all assets
    benchmark : str
        Benchmark ticker (e.g. "SPY")
    start_date : pd.Timestamp
        Start date for the benchmark simulation
    initial_nav : float
        Starting value (should match portfolio initial_nav)

    Returns
    -------
    pd.DataFrame with columns: nav, daily_return
    """
    bench_returns = simple_returns.loc[start_date:, benchmark]

    nav = initial_nav
    records = []

    for date, ret in bench_returns.items():
        nav = nav * (1 + ret)
        records.append({"date": date, "nav": nav, "daily_return": ret})

    result = pd.DataFrame(records).set_index("date")
    logger.info(
        f"Benchmark simulation complete | "
        f"Final NAV: ${result['nav'].iloc[-1]:,.2f}"
    )
    return result
