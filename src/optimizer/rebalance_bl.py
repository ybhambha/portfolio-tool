"""
rebalance_bl.py
---------------
Walk-forward rebalancing using Black-Litterman expected returns.

At each rebalance date:
  1. Get training window (past 252 days of returns)
  2. Compute covariance matrix (Ledoit-Wolf)
  3. Compute time-varying SPY weights using price-relative approach
  4. Compute equilibrium returns from those SPY weights
  5. Run BL model (with views if available, otherwise equilibrium only)
  6. Run MVO optimizer with BL expected returns
  7. Record new target weights

This is identical to the original walk-forward in rebalance.py
except step 3-5 replace the historical mean with BL returns.
"""

import pandas as pd
import numpy as np
import logging

from src.optimizer.covariance import compute_covariance, compute_expected_returns
from src.optimizer.optimizer import optimize_portfolio
from src.optimizer.rebalance import get_rebalance_dates
from src.optimizer.spy_weights import compute_spy_weights_at_date
from src.optimizer.black_litterman import (
    compute_equilibrium_returns,
    black_litterman,
    build_views,
)

logger = logging.getLogger(__name__)


def run_walk_forward_bl(
    log_returns: pd.DataFrame,
    prices: pd.DataFrame,
    etfs: list[str],
    lookback_days: int = 252,
    rebalance_frequency: str = "monthly",
    weight_min: float = 0.0,
    weight_max: float = 0.30,
    transaction_cost_bps: float = 5.0,
    risk_aversion: float = 2.5,
    risk_free_rate: float = 0.05,
    cov_method: str = "ledoit_wolf",
    tau: float = 0.05,
    views_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Walk-forward optimization using Black-Litterman expected returns.

    At each rebalance date the equilibrium is computed from
    time-varying SPY sector weights using a rolling 252-day
    price-relative approach — no look-ahead bias.

    Parameters
    ----------
    log_returns       : daily log returns, shape (days, tickers)
    prices            : daily prices, shape (days, tickers)
    etfs              : list of ETF tickers
    lookback_days     : rolling window for covariance and SPY weights
    rebalance_frequency: "monthly" or "quarterly"
    weight_min        : minimum weight per ETF
    weight_max        : maximum weight per ETF
    transaction_cost_bps: cost per trade in basis points
    risk_aversion     : BL risk aversion parameter (default 2.5)
    risk_free_rate    : annualized risk-free rate
    cov_method        : covariance method ("ledoit_wolf" or "sample")
    tau               : BL scalar (trust in equilibrium)
    views_df          : optional views DataFrame for BL
                        must have columns: implied_return, analyst_count

    Returns
    -------
    pd.DataFrame : target weights at each rebalance date
                   shape (n_dates, n_etfs)
    """
    etf_returns = log_returns[etfs]
    etf_prices  = prices[etfs]

    rebalance_dates = get_rebalance_dates(
        etf_returns.index, frequency=rebalance_frequency
    )

    # Only keep dates where we have enough lookback data
    first_valid    = etf_returns.index[lookback_days]
    rebalance_dates = [d for d in rebalance_dates if d >= first_valid]

    logger.info(
        f"BL Walk-forward | {len(rebalance_dates)} periods | "
        f"frequency={rebalance_frequency} | lookback={lookback_days}"
    )

    all_weights     = {}
    current_weights = None
    bl_used_count   = 0
    eq_used_count   = 0

    for i, rebal_date in enumerate(rebalance_dates):

        # --- Training window ---
        window_returns = etf_returns.loc[:rebal_date].tail(lookback_days)

        if len(window_returns) < lookback_days * 0.8:
            logger.warning(
                f"Skipping {rebal_date} — insufficient data"
            )
            continue

        # --- Step 1: Covariance matrix ---
        cov = compute_covariance(
            window_returns, method=cov_method, annualize=True
        )

        # --- Step 2: Time-varying SPY weights ---
        spy_weights = compute_spy_weights_at_date(
            prices=etf_prices,
            rebalance_date=rebal_date,
            tickers=etfs,
            lookback_days=lookback_days,
        )

        # --- Step 3: Equilibrium returns from SPY weights ---
        pi = compute_equilibrium_returns(
            market_weights=spy_weights,
            cov_matrix=cov,
            risk_aversion=risk_aversion,
        )

        # --- Step 4: BL posterior (or equilibrium if no views) ---
        if views_df is not None:
            P, Q, Omega = build_views(
                analyst_data=views_df,
                tickers=etfs,
                min_analysts=1,
            )
            if P is not None:
                mu = black_litterman(pi, cov, P, Q, Omega, tau)
                bl_used_count += 1
            else:
                mu = pi
                eq_used_count += 1
        else:
            mu = pi
            eq_used_count += 1

        # --- Step 5: Optimize ---
        optimal_weights = optimize_portfolio(
            mu=mu,
            cov=cov,
            current_weights=current_weights,
            weight_min=weight_min,
            weight_max=weight_max,
            transaction_cost_bps=transaction_cost_bps,
            risk_aversion=1.0,
        )

        all_weights[rebal_date] = optimal_weights
        current_weights = optimal_weights

        if i % 12 == 0:
            logger.info(
                f"BL Rebalance {i+1}/{len(rebalance_dates)}: "
                f"{rebal_date.date()} | "
                f"XLK={optimal_weights.get('XLK', 0):.1%}"
            )

    weights_df = pd.DataFrame(all_weights).T
    weights_df.index.name = "rebalance_date"

    logger.info(
        f"BL Walk-forward complete | "
        f"{len(weights_df)} periods | "
        f"BL used: {bl_used_count} | "
        f"Equilibrium used: {eq_used_count}"
    )

    return weights_df
