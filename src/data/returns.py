"""
returns.py
----------
Computes log and simple returns from a cleaned price DataFrame.

  - Log returns  : used for covariance matrix and MVO optimizer
  - Simple returns: used for NAV simulation in the backtester

Both are stored and passed explicitly to downstream modules to avoid
silent compounding errors from mixing the two.
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns: r_t = ln(P_t / P_{t-1})

    Log returns are time-additive and better behaved statistically.
    Use these for the covariance matrix and MVO optimizer.

    Parameters
    ----------
    prices : aligned price DataFrame, shape (days, tickers)

    Returns
    -------
    pd.DataFrame : log returns, shape (days-1, tickers)
    """
    log_returns = np.log(prices / prices.shift(1)).dropna()
    logger.info(f"Computed log returns: shape {log_returns.shape}")
    return log_returns


def compute_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily simple returns: r_t = (P_t - P_{t-1}) / P_{t-1}

    Use these for NAV simulation:  NAV_t = NAV_{t-1} * (1 + r_t)

    Parameters
    ----------
    prices : aligned price DataFrame, shape (days, tickers)

    Returns
    -------
    pd.DataFrame : simple returns, shape (days-1, tickers)
    """
    simple_returns = prices.pct_change().dropna()
    logger.info(f"Computed simple returns: shape {simple_returns.shape}")
    return simple_returns


def returns_summary(log_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Print annualized return and volatility for each ticker.
    Useful sanity-check before running the optimizer.
    """
    summary = pd.DataFrame({
        "ann_return_pct": (log_returns.mean() * 252 * 100).round(2),
        "ann_vol_pct"   : (log_returns.std() * np.sqrt(252) * 100).round(2),
        "sharpe_approx" : (
            (log_returns.mean() * 252) /
            (log_returns.std() * np.sqrt(252))
        ).round(3),
    })
    print("\n--- Returns Summary (annualized) ---")
    print(summary.to_string())
    print("-------------------------------------\n")
    return summary
