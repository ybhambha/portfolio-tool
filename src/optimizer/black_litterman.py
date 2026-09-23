"""
black_litterman.py
------------------
Implements the Black-Litterman model for expected return estimation.

The Black-Litterman model blends two sources of information:
  1. Market equilibrium returns (implied by SPY sector weights)
  2. Investor views (from flow data or your own forecasts)

The result is a set of "posterior" expected returns that are more
stable and realistic than pure historical means.

Formula:
  mu_BL = [(tau*Sigma)^-1 + P^T * Omega^-1 * P]^-1
          * [(tau*Sigma)^-1 * pi + P^T * Omega^-1 * Q]

Where:
  pi    = equilibrium returns (from SPY sector weights)
  P     = pick matrix (which assets each view is about)
  Q     = view returns (what you expect)
  Omega = view uncertainty matrix (how confident you are)
  tau   = scalar (how much to trust equilibrium vs views)
"""

import numpy as np
import pandas as pd
import yfinance as yf
import logging

from src.optimizer.spy_weights import fetch_spy_sector_weights

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1: Compute market equilibrium returns (pi)
# ---------------------------------------------------------------------------

def compute_market_cap_weights(tickers: list[str]) -> pd.Series:
    """
    Fetch SPY's actual sector weights programmatically.
    Uses a fallback chain: SPDR website → yfinance → hardcoded.

    Parameters
    ----------
    tickers : list of ETF tickers

    Returns
    -------
    pd.Series : SPY sector weights, indexed by ticker
    """
    return fetch_spy_sector_weights(tickers)


def compute_equilibrium_returns(
    market_weights: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_aversion: float = 2.5,
) -> pd.Series:
    """
    Compute implied equilibrium returns using reverse optimization.

    pi = lambda * Sigma * w_market

    This answers: "What expected returns would make an investor
    hold the current SPY sector weights as optimal?"

    Parameters
    ----------
    market_weights : SPY sector weights from compute_market_cap_weights()
    cov_matrix     : annualized covariance matrix
    risk_aversion  : market risk aversion coefficient (typically 2-3)

    Returns
    -------
    pd.Series : equilibrium expected returns, one per asset
    """
    w  = market_weights.reindex(cov_matrix.index).fillna(0).values
    pi = risk_aversion * cov_matrix.values @ w
    return pd.Series(pi, index=cov_matrix.index)


# ---------------------------------------------------------------------------
# Step 2: Fetch analyst views (kept for future use)
# ---------------------------------------------------------------------------

def fetch_analyst_views(tickers: list[str]) -> pd.DataFrame:
    """
    Fetch analyst price targets from yfinance.
    Note: sector ETFs typically have no analyst coverage.
    This function is kept for future use with individual stocks.
    """
    rows = []

    for ticker in tickers:
        try:
            t    = yf.Ticker(ticker)
            info = t.info

            current_price = info.get("regularMarketPrice") or \
                           info.get("currentPrice") or \
                           info.get("navPrice")

            target_mean = info.get("targetMeanPrice")
            target_low  = info.get("targetLowPrice")
            target_high = info.get("targetHighPrice")
            n_analysts  = info.get("numberOfAnalystOpinions", 0)
            recommend   = info.get("recommendationKey", "none")

            if current_price and target_mean:
                implied_return = (target_mean - current_price) / current_price
            else:
                implied_return = None

            rows.append({
                "ticker"         : ticker,
                "current_price"  : round(current_price, 2) if current_price else None,
                "target_mean"    : round(target_mean, 2) if target_mean else None,
                "target_low"     : round(target_low, 2) if target_low else None,
                "target_high"    : round(target_high, 2) if target_high else None,
                "implied_return" : round(implied_return, 4) if implied_return else None,
                "analyst_count"  : n_analysts,
                "recommendation" : recommend,
            })

        except Exception as e:
            logger.warning(f"Could not fetch analyst data for {ticker}: {e}")
            rows.append({
                "ticker"         : ticker,
                "current_price"  : None,
                "target_mean"    : None,
                "target_low"     : None,
                "target_high"    : None,
                "implied_return" : None,
                "analyst_count"  : 0,
                "recommendation" : "none",
            })

    df = pd.DataFrame(rows).set_index("ticker")
    return df


# ---------------------------------------------------------------------------
# Step 3: Build views matrix
# ---------------------------------------------------------------------------

def build_views(
    analyst_data: pd.DataFrame,
    tickers: list[str],
    min_analysts: int = 3,
    confidence_scale: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the Black-Litterman views matrices P, Q, Omega
    from analyst data or flow-based views.

    Parameters
    ----------
    analyst_data     : DataFrame with implied_return and analyst_count
    tickers          : list of ETF tickers
    min_analysts     : minimum coverage to include a view
    confidence_scale : scales view uncertainty

    Returns
    -------
    P     : pick matrix, shape (n_views, n_assets)
    Q     : view returns, shape (n_views,)
    Omega : view uncertainty, shape (n_views, n_views) diagonal
    """
    valid_views = analyst_data[
        (analyst_data["implied_return"].notna()) &
        (analyst_data["analyst_count"] >= min_analysts)
    ]

    if len(valid_views) == 0:
        logger.warning("No valid views found")
        return None, None, None

    n_assets = len(tickers)
    n_views  = len(valid_views)

    P     = np.zeros((n_views, n_assets))
    Q     = np.zeros(n_views)
    omega = np.zeros(n_views)

    for i, (ticker, row) in enumerate(valid_views.iterrows()):
        if ticker in tickers:
            asset_idx       = tickers.index(ticker)
            P[i, asset_idx] = 1.0
            Q[i]            = row["implied_return"]
            n               = max(row["analyst_count"], 1)
            omega[i]        = confidence_scale / n

    Omega = np.diag(omega)
    logger.info(f"Built {n_views} views")
    return P, Q, Omega


# ---------------------------------------------------------------------------
# Step 4: Black-Litterman posterior
# ---------------------------------------------------------------------------

def black_litterman(
    pi: pd.Series,
    cov_matrix: pd.DataFrame,
    P: np.ndarray,
    Q: np.ndarray,
    Omega: np.ndarray,
    tau: float = 0.05,
) -> pd.Series:
    """
    Compute Black-Litterman posterior expected returns.

    Parameters
    ----------
    pi         : equilibrium returns
    cov_matrix : annualized covariance matrix
    P          : pick matrix
    Q          : view returns
    Omega      : view uncertainty matrix
    tau        : scalar controlling trust in equilibrium

    Returns
    -------
    pd.Series : posterior expected returns
    """
    tickers       = cov_matrix.index.tolist()
    Sigma         = cov_matrix.values
    pi_arr        = pi.reindex(tickers).values
    tau_sigma_inv = np.linalg.inv(tau * Sigma)
    omega_inv     = np.linalg.inv(Omega)
    M             = tau_sigma_inv + P.T @ omega_inv @ P
    mu_bl         = np.linalg.inv(M) @ (
                        tau_sigma_inv @ pi_arr +
                        P.T @ omega_inv @ Q
                    )

    logger.info("Black-Litterman posterior returns computed")
    return pd.Series(mu_bl, index=tickers)


# ---------------------------------------------------------------------------
# Main function: compute BL expected returns
# ---------------------------------------------------------------------------

def compute_bl_expected_returns(
    tickers: list[str],
    cov_matrix: pd.DataFrame,
    risk_aversion: float = 2.5,
    tau: float = 0.05,
    min_analysts: int = 3,
    confidence_scale: float = 0.5,
    views_df: pd.DataFrame | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Full Black-Litterman pipeline.

    Parameters
    ----------
    tickers          : list of ETF tickers
    cov_matrix       : annualized covariance matrix
    risk_aversion    : market risk aversion (default 2.5)
    tau              : equilibrium trust scalar (default 0.05)
    min_analysts     : min coverage for a view
    confidence_scale : view uncertainty scaling
    views_df         : optional pre-built views DataFrame
                       (e.g. from flow-based views)
                       must have columns: implied_return, analyst_count

    Returns
    -------
    mu_bl        : pd.Series of BL expected returns
    analyst_data : pd.DataFrame of raw views data
    """

    # Step 1: Get SPY sector weights programmatically
    market_weights = compute_market_cap_weights(tickers)

    # Step 2: Compute equilibrium returns
    print("Computing equilibrium returns...")
    pi = compute_equilibrium_returns(
        market_weights, cov_matrix, risk_aversion
    )

    # Step 3: Get views — use provided views_df or fetch analyst data
    if views_df is not None:
        print("Using provided views (e.g. flow-based)...")
        analyst_data = views_df
    else:
        print("Fetching analyst price targets...")
        analyst_data = fetch_analyst_views(tickers)

    # Step 4: Build views matrices
    print("Building views matrices...")
    P, Q, Omega = build_views(
        analyst_data, tickers, min_analysts, confidence_scale
    )

    # Step 5: If no views available → return equilibrium (Fix 1)
    if P is None:
        logger.warning("No views available — returning equilibrium returns")
        print("No views available — using market equilibrium returns")
        return pi, analyst_data

    # Step 6: Compute BL posterior
    print("Computing Black-Litterman posterior...")
    mu_bl = black_litterman(pi, cov_matrix, P, Q, Omega, tau)

    return mu_bl, analyst_data


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def print_bl_diagnostics(
    pi: pd.Series,
    mu_historical: pd.Series,
    mu_bl: pd.Series,
    analyst_data: pd.DataFrame,
) -> None:
    """Print comparison of equilibrium, historical, and BL returns."""
    print("\n--- Black-Litterman Diagnostics ---")
    print(f"{'ETF':<6} {'Equilib%':>10} {'Hist%':>10} "
          f"{'BL%':>10} {'View Return':>12} {'Coverage':>10}")
    print("-" * 65)

    for ticker in mu_bl.index:
        eq_ret   = pi.get(ticker, 0) * 100
        hist_ret = mu_historical.get(ticker, 0) * 100
        bl_ret   = mu_bl[ticker] * 100

        if ticker in analyst_data.index:
            implied = analyst_data.loc[ticker, "implied_return"]
            n       = analyst_data.loc[ticker, "analyst_count"]
            view_s  = f"{implied*100:.1f}%" if implied else "N/A"
            n_s     = str(int(n)) if n else "0"
        else:
            view_s = "N/A"
            n_s    = "0"

        print(f"{ticker:<6} {eq_ret:>9.2f}% {hist_ret:>9.2f}% "
              f"{bl_ret:>9.2f}% {view_s:>12} {n_s:>10}")

    print("-" * 65)
    print(f"{'Mean':<6} {pi.mean()*100:>9.2f}% "
          f"{mu_historical.mean()*100:>9.2f}% "
          f"{mu_bl.mean()*100:>9.2f}%")
    print("-----------------------------------\n")
