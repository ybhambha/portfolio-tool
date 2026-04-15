"""
optimizer.py
------------
Mean-Variance Optimization (MVO) engine using cvxpy.

Solves the classic Markowitz portfolio optimization problem:
  Maximize: w^T * mu - (lambda/2) * w^T * Sigma * w - transaction_cost
  Subject to:
    - sum(w) = 1          (fully invested)
    - w >= weight_min     (long-only by default)
    - w <= weight_max     (concentration limit)

Transaction costs are baked directly into the objective function,
penalizing turnover proportional to the cost in basis points.
This produces naturally lower-turnover portfolios compared to
applying costs post-hoc.
"""

import numpy as np
import pandas as pd
import cvxpy as cp
import logging

logger = logging.getLogger(__name__)


def optimize_portfolio(
    mu: pd.Series,
    cov: pd.DataFrame,
    current_weights: pd.Series | None = None,
    weight_min: float = 0.0,
    weight_max: float = 0.40,
    transaction_cost_bps: float = 5.0,
    risk_aversion: float = 1.0,
) -> pd.Series:
    """
    Solve the MVO problem and return optimal portfolio weights.

    Parameters
    ----------
    mu : pd.Series
        Annualized expected returns, one per asset
    cov : pd.DataFrame
        Annualized covariance matrix, shape (n_assets, n_assets)
    current_weights : pd.Series or None
        Current portfolio weights before rebalancing.
        If None (first period), assumes equal weights.
    weight_min : float
        Minimum weight per asset (0.0 = long-only)
    weight_max : float
        Maximum weight per asset (e.g. 0.40 = max 40% in any one ETF)
    transaction_cost_bps : float
        Cost per trade in basis points (5 bps = 0.05%)
    risk_aversion : float
        Lambda — higher values penalize variance more.
        1.0 is a reasonable default. Increase to get more conservative portfolios.

    Returns
    -------
    pd.Series : optimal weights, indexed by ticker
    """
    tickers = mu.index.tolist()
    n = len(tickers)

    mu_arr = mu.values
    cov_arr = cov.values

    # Default to equal weights if no current weights provided
    if current_weights is None:
        w0 = np.ones(n) / n
    else:
        w0 = current_weights.reindex(tickers).fillna(0.0).values

    # Decision variable: portfolio weights
    w = cp.Variable(n)

    # Transaction cost penalty: cost_bps/10000 * sum(|w - w0|)
    cost_per_unit = transaction_cost_bps / 10_000
    turnover = cp.norm1(w - w0)  # L1 norm = sum of absolute weight changes
    transaction_cost = cost_per_unit * turnover

    # Objective: maximize risk-adjusted return minus transaction costs
    portfolio_return = mu_arr @ w
    portfolio_variance = cp.quad_form(w, cov_arr)
    objective = cp.Maximize(
        portfolio_return
        - (risk_aversion / 2) * portfolio_variance
        - transaction_cost
    )

    # Constraints
    constraints = [
        cp.sum(w) == 1,          # fully invested
        w >= weight_min,          # long-only (or allow shorting if weight_min < 0)
        w <= weight_max,          # concentration limit
    ]

    # Solve
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.CLARABEL, warm_start=True)

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        logger.warning(f"Optimizer status: {problem.status} — falling back to equal weights")
        return pd.Series(np.ones(n) / n, index=tickers)

    optimal_weights = pd.Series(w.value, index=tickers)

    # Clean up tiny numerical noise (weights below 0.001% → 0)
    optimal_weights = optimal_weights.clip(lower=0)
    optimal_weights = optimal_weights / optimal_weights.sum()  # renormalize

    logger.info(
        f"Optimization solved | status={problem.status} | "
        f"active positions={( optimal_weights > 0.001).sum()}"
    )

    return optimal_weights


def compute_portfolio_metrics(
    weights: pd.Series,
    mu: pd.Series,
    cov: pd.DataFrame,
    risk_free_rate: float = 0.05,
) -> dict:
    """
    Compute expected portfolio metrics given optimal weights.

    Parameters
    ----------
    weights : pd.Series
        Portfolio weights
    mu : pd.Series
        Annualized expected returns
    cov : pd.DataFrame
        Annualized covariance matrix
    risk_free_rate : float
        Annualized risk-free rate for Sharpe ratio

    Returns
    -------
    dict with expected_return, expected_volatility, sharpe_ratio
    """
    w = weights.values
    expected_return = float(mu.values @ w)
    expected_volatility = float(np.sqrt(w @ cov.values @ w))
    sharpe = (expected_return - risk_free_rate) / expected_volatility

    return {
        "expected_return": round(expected_return * 100, 2),       # as %
        "expected_volatility": round(expected_volatility * 100, 2),  # as %
        "sharpe_ratio": round(sharpe, 3),
    }


def print_weights(weights: pd.Series, title: str = "Portfolio Weights") -> None:
    """Pretty-print portfolio weights sorted by allocation."""
    print(f"\n--- {title} ---")
    sorted_w = weights.sort_values(ascending=False)
    for ticker, w in sorted_w.items():
        bar = "█" * int(w * 40)
        print(f"  {ticker:<6} {w*100:>6.2f}%  {bar}")
    print(f"  {'TOTAL':<6} {weights.sum()*100:>6.2f}%")
    print("-----------------------------\n")
