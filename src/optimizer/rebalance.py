"""
rebalance.py
------------
Implements the rolling walk-forward rebalancing loop.

At each rebalance date:
  1. Gather the past `lookback_days` of log returns (training window)
  2. Estimate covariance matrix and expected returns on that window
  3. Run the MVO optimizer with current weights and transaction costs
  4. Record the new target weights

This is a strict walk-forward approach — the optimizer never sees
future data, eliminating look-ahead bias.
"""

import pandas as pd
import numpy as np
import logging
from dateutil.relativedelta import relativedelta

from src.optimizer.covariance import compute_covariance, compute_expected_returns
from src.optimizer.optimizer import optimize_portfolio

logger = logging.getLogger(__name__)


def get_rebalance_dates(
    index: pd.DatetimeIndex,
    frequency: str = "monthly",
) -> list[pd.Timestamp]:
    """
    Generate rebalance dates from a DatetimeIndex.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Full trading day index from the price data
    frequency : str
        "monthly" or "quarterly"

    Returns
    -------
    list of Timestamps — last trading day of each period
    """
    if frequency == "monthly":
        # Last trading day of each month
        return list(
            pd.Series(index).groupby(
                pd.Series(index).dt.to_period("M")
            ).last()
        )
    elif frequency == "quarterly":
        return list(
            pd.Series(index).groupby(
                pd.Series(index).dt.to_period("Q")
            ).last()
        )
    else:
        raise ValueError(f"Unknown frequency: {frequency}. Use 'monthly' or 'quarterly'.")


def run_walk_forward(
    log_returns: pd.DataFrame,
    etfs: list[str],
    lookback_days: int = 252,
    rebalance_frequency: str = "monthly",
    weight_min: float = 0.0,
    weight_max: float = 0.40,
    transaction_cost_bps: float = 5.0,
    risk_aversion: float = 1.0,
    risk_free_rate: float = 0.05,
    cov_method: str = "ledoit_wolf",
) -> pd.DataFrame:
    """
    Run the full walk-forward optimization loop.

    At each rebalance date, fits the optimizer on the past `lookback_days`
    of returns only — never using future data.

    Parameters
    ----------
    log_returns : pd.DataFrame
        Daily log returns for all assets including benchmark
    etfs : list[str]
        List of ETF tickers to include in the portfolio (excludes benchmark)
    lookback_days : int
        Number of trading days to use for covariance estimation
    rebalance_frequency : str
        "monthly" or "quarterly"
    weight_min, weight_max : float
        Weight bounds per asset
    transaction_cost_bps : float
        Transaction cost per trade in basis points
    risk_aversion : float
        Risk aversion parameter for the optimizer
    risk_free_rate : float
        Annualized risk-free rate
    cov_method : str
        Covariance estimation method: "ledoit_wolf" or "sample"

    Returns
    -------
    pd.DataFrame : target weights at each rebalance date, shape (n_dates, n_etfs)
    """
    # Use only ETF returns (exclude benchmark from optimization)
    etf_returns = log_returns[etfs]

    rebalance_dates = get_rebalance_dates(etf_returns.index, frequency=rebalance_frequency)

    # Only keep rebalance dates where we have enough lookback data
    first_valid_date = etf_returns.index[lookback_days]
    rebalance_dates = [d for d in rebalance_dates if d >= first_valid_date]

    logger.info(
        f"Walk-forward rebalancing | {len(rebalance_dates)} periods | "
        f"frequency={rebalance_frequency} | lookback={lookback_days} days"
    )

    all_weights = {}
    current_weights = None  # Equal weights on first rebalance

    for i, rebal_date in enumerate(rebalance_dates):
        # Training window: past `lookback_days` trading days up to rebal_date
        window = etf_returns.loc[:rebal_date].tail(lookback_days)

        if len(window) < lookback_days * 0.8:
            logger.warning(f"Skipping {rebal_date} — insufficient data ({len(window)} days)")
            continue

        # Estimate inputs
        mu = compute_expected_returns(window, annualize=True)
        cov = compute_covariance(window, method=cov_method, annualize=True)

        # Optimize
        optimal_weights = optimize_portfolio(
            mu=mu,
            cov=cov,
            current_weights=current_weights,
            weight_min=weight_min,
            weight_max=weight_max,
            transaction_cost_bps=transaction_cost_bps,
            risk_aversion=risk_aversion,
        )

        all_weights[rebal_date] = optimal_weights
        current_weights = optimal_weights

        if i % 12 == 0:  # log progress every 12 periods
            logger.info(f"Rebalance {i+1}/{len(rebalance_dates)}: {rebal_date.date()}")

    weights_df = pd.DataFrame(all_weights).T
    weights_df.index.name = "rebalance_date"

    logger.info(f"Walk-forward complete | {len(weights_df)} rebalance periods")

    return weights_df
