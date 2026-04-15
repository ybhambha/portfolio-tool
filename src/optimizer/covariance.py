"""
covariance.py
-------------
Estimates the covariance matrix of asset returns.

Two methods are supported:
  - "sample"       : standard sample covariance matrix
  - "ledoit_wolf"  : shrinkage estimator (recommended for small universes)

Why Ledoit-Wolf?
With only 11 ETFs and ~252 days of data, the sample covariance matrix can be
noisy and poorly conditioned. Ledoit-Wolf shrinks the sample covariance toward
a structured target (scaled identity), producing a more stable estimate that
leads to better out-of-sample portfolio performance.
"""

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
import logging

logger = logging.getLogger(__name__)


def compute_covariance(
    log_returns: pd.DataFrame,
    method: str = "ledoit_wolf",
    annualize: bool = True,
) -> pd.DataFrame:
    """
    Estimate the covariance matrix of asset returns.

    Parameters
    ----------
    log_returns : pd.DataFrame
        Daily log returns, shape (days, n_assets)
    method : str
        "ledoit_wolf" (recommended) or "sample"
    annualize : bool
        If True, multiply by 252 to annualize

    Returns
    -------
    pd.DataFrame : covariance matrix, shape (n_assets, n_assets)
    """
    tickers = log_returns.columns.tolist()
    returns_array = log_returns.values

    if method == "ledoit_wolf":
        lw = LedoitWolf()
        lw.fit(returns_array)
        cov_matrix = lw.covariance_
        logger.info(f"Ledoit-Wolf shrinkage coefficient: {lw.shrinkage_:.4f}")
    elif method == "sample":
        cov_matrix = np.cov(returns_array, rowvar=False)
    else:
        raise ValueError(f"Unknown covariance method: {method}. Use 'ledoit_wolf' or 'sample'.")

    if annualize:
        cov_matrix = cov_matrix * 252

    cov_df = pd.DataFrame(cov_matrix, index=tickers, columns=tickers)
    logger.info(f"Covariance matrix computed | method={method} | shape={cov_df.shape}")

    return cov_df


def compute_expected_returns(
    log_returns: pd.DataFrame,
    annualize: bool = True,
) -> pd.Series:
    """
    Compute expected returns as the historical mean of log returns.

    Parameters
    ----------
    log_returns : pd.DataFrame
        Daily log returns, shape (days, n_assets)
    annualize : bool
        If True, multiply by 252 to annualize

    Returns
    -------
    pd.Series : expected returns, one per asset
    """
    mu = log_returns.mean()
    if annualize:
        mu = mu * 252
    logger.info(f"Expected returns computed | annualized={annualize}")
    return mu


def covariance_diagnostics(cov_df: pd.DataFrame) -> None:
    """
    Print diagnostics to check if the covariance matrix is well-conditioned.
    A condition number above 1000 suggests instability in the optimizer.
    """
    eigenvalues = np.linalg.eigvalsh(cov_df.values)
    condition_number = eigenvalues.max() / eigenvalues.min()

    print("\n--- Covariance Matrix Diagnostics ---")
    print(f"Shape          : {cov_df.shape}")
    print(f"Min eigenvalue : {eigenvalues.min():.6f}")
    print(f"Max eigenvalue : {eigenvalues.max():.6f}")
    print(f"Condition number: {condition_number:.1f}")
    if condition_number > 1000:
        print("WARNING: High condition number — consider using Ledoit-Wolf shrinkage")
    else:
        print("Condition number looks healthy")
    print("-------------------------------------\n")
