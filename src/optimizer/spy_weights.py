"""
spy_weights.py
--------------
Computes time-varying SPY sector weights using a rolling 252-day
price-relative approach.

At each rebalance date T:
  1. Look back 252 trading days to get base_date (T - 252 days)
  2. Get ETF prices at base_date and T
  3. Compute price relatives: price(T) / price(base_date)
  4. Update known base weights proportionally
  5. Renormalize to sum to 1.0

This approach:
  - Uses only data available at each point in time (no look-ahead bias)
  - Updates monthly reflecting actual market movements
  - Is consistent with the 252-day lookback used everywhere else
  - Requires no external data source beyond prices already in our pipeline

Why price relatives work:
  If XLK outperforms XLE over the past year, XLK's share of the
  total market cap grows relative to XLE. This mirrors exactly how
  SPY's sector weights evolve as stock prices change.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Known base weights — SPY sector composition at start of backtest
# Source: SPDR Holdings, June 2018
# These are only used as the starting proportions, not fixed values
# ---------------------------------------------------------------------------

SPY_BASE_WEIGHTS_JUNE_2018 = {
    "XLK" : 0.210,   # Technology
    "XLV" : 0.150,   # Health Care
    "XLF" : 0.140,   # Financials
    "XLY" : 0.100,   # Consumer Discretionary
    "XLC" : 0.100,   # Communication Services (created Sept 2018)
    "XLI" : 0.095,   # Industrials
    "XLP" : 0.075,   # Consumer Staples
    "XLE" : 0.060,   # Energy
    "XLB" : 0.030,   # Materials
    "XLRE": 0.030,   # Real Estate
    "XLU" : 0.030,   # Utilities
}


# ---------------------------------------------------------------------------
# Core function: compute SPY weights at a single rebalance date
# ---------------------------------------------------------------------------

def compute_spy_weights_at_date(
    prices: pd.DataFrame,
    rebalance_date: pd.Timestamp,
    tickers: list[str],
    lookback_days: int = 252,
) -> pd.Series:
    """
    Compute approximate SPY sector weights at a given rebalance date
    using a rolling 252-day price-relative approach.

    Steps:
      1. Find base_date = rebalance_date - lookback_days trading days
      2. Get ETF prices at base_date and rebalance_date
      3. Compute price relatives
      4. Update base weights proportionally
      5. Renormalize to sum to 1.0

    Parameters
    ----------
    prices         : aligned price DataFrame from data pipeline
                     shape (trading_days, tickers)
    rebalance_date : the date for which to compute weights
    tickers        : list of ETF tickers
    lookback_days  : number of trading days to look back (default 252)

    Returns
    -------
    pd.Series : SPY sector weights at rebalance_date, indexed by ticker
    """

    # Get all trading days up to and including rebalance_date
    available_dates = prices.index[prices.index <= rebalance_date]

    if len(available_dates) < lookback_days:
        logger.warning(
            f"Insufficient data at {rebalance_date.date()} — "
            f"only {len(available_dates)} days available, "
            f"need {lookback_days}. Using base weights."
        )
        base = pd.Series(
            {t: SPY_BASE_WEIGHTS_JUNE_2018.get(t, 0.0) for t in tickers}
        )
        return base / base.sum()

    # Base date: lookback_days trading days before rebalance_date
    base_date = available_dates[-lookback_days]

    # Get prices at both dates
    base_prices  = prices.loc[base_date,  tickers]
    curr_prices  = prices.loc[rebalance_date, tickers]

    # Compute price relatives: how much each ETF moved over the window
    price_relatives = curr_prices / base_prices

    # Handle any missing or zero base prices
    price_relatives = price_relatives.replace([np.inf, -np.inf], np.nan)
    price_relatives = price_relatives.fillna(1.0)  # neutral if missing

    # Get known base weights for these tickers
    base_weights = pd.Series(
        {t: SPY_BASE_WEIGHTS_JUNE_2018.get(t, 0.0) for t in tickers}
    )

    # Update weights proportionally to price performance
    updated_weights = base_weights * price_relatives

    # Renormalize to sum to 1.0
    total = updated_weights.sum()
    if total <= 0:
        logger.warning("Zero total weight — returning equal weights")
        return pd.Series(np.ones(len(tickers)) / len(tickers), index=tickers)

    spy_weights = updated_weights / total

    logger.debug(
        f"SPY weights at {rebalance_date.date()} | "
        f"base_date={base_date.date()} | "
        f"XLK={spy_weights.get('XLK', 0):.1%} | "
        f"XLE={spy_weights.get('XLE', 0):.1%}"
    )

    return spy_weights


# ---------------------------------------------------------------------------
# Batch function: compute SPY weights for all rebalance dates
# ---------------------------------------------------------------------------

def compute_all_spy_weights(
    prices: pd.DataFrame,
    rebalance_dates: list[pd.Timestamp],
    tickers: list[str],
    lookback_days: int = 252,
) -> pd.DataFrame:
    """
    Compute SPY sector weights for every rebalance date in the backtest.

    Parameters
    ----------
    prices          : aligned price DataFrame from data pipeline
    rebalance_dates : list of monthly rebalance dates
    tickers         : list of ETF tickers
    lookback_days   : rolling window size (default 252)

    Returns
    -------
    pd.DataFrame : SPY weights at each rebalance date
                   shape (n_rebalance_dates, n_tickers)
                   index = rebalance_dates
    """
    all_weights = {}

    for rebal_date in rebalance_dates:
        weights = compute_spy_weights_at_date(
            prices=prices,
            rebalance_date=rebal_date,
            tickers=tickers,
            lookback_days=lookback_days,
        )
        all_weights[rebal_date] = weights

    weights_df = pd.DataFrame(all_weights).T
    weights_df.index.name = "rebalance_date"

    logger.info(
        f"Computed SPY weights for {len(weights_df)} rebalance dates"
    )

    return weights_df


# ---------------------------------------------------------------------------
# Diagnostics: print how weights evolved over time
# ---------------------------------------------------------------------------

def print_spy_weight_evolution(
    weights_df: pd.DataFrame,
    tickers: list[str] = None,
    sample_dates: int = 5,
) -> None:
    """
    Print how SPY sector weights evolved across the backtest period.

    Parameters
    ----------
    weights_df   : output from compute_all_spy_weights()
    tickers      : which tickers to show (default: all)
    sample_dates : how many dates to sample for display
    """
    if tickers is None:
        tickers = weights_df.columns.tolist()

    # Sample evenly spaced dates
    indices  = np.linspace(0, len(weights_df) - 1, sample_dates, dtype=int)
    sampled  = weights_df.iloc[indices]

    print("\n--- SPY Sector Weight Evolution ---")
    print(f"{'ETF':<6}", end="")
    for date in sampled.index:
        print(f"  {date.strftime('%b %Y'):>10}", end="")
    print()
    print("-" * (6 + 12 * sample_dates))

    for ticker in tickers:
        print(f"{ticker:<6}", end="")
        for date in sampled.index:
            w = sampled.loc[date, ticker]
            print(f"  {w*100:>9.2f}%", end="")
        print()

    print("-" * (6 + 12 * sample_dates))
    print(f"{'TOTAL':<6}", end="")
    for date in sampled.index:
        print(f"  {sampled.loc[date].sum()*100:>9.2f}%", end="")
    print("\n")


# ---------------------------------------------------------------------------
# Live signal: fetch current SPY weights (for non-backtest use)
# ---------------------------------------------------------------------------

def fetch_spy_sector_weights(tickers: list[str]) -> pd.Series:
    """
    For live signal generation (not backtesting):
    Compute current SPY weights using the most recent available prices.

    This fetches the latest 252 days of prices from yfinance
    and applies the same price-relative approach.

    Parameters
    ----------
    tickers : list of ETF tickers

    Returns
    -------
    pd.Series : current SPY sector weights
    """
    import yfinance as yf

    print("Fetching latest prices for SPY weight calculation...")

    # Fetch recent prices
    raw = yf.download(
        tickers,
        period="400d",
        auto_adjust=False,
        progress=False,
    )

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Adj Close"]
    else:
        prices = raw[["Adj Close"]].rename(
            columns={"Adj Close": tickers[0]}
        )

    prices = prices.dropna()

    if len(prices) < 252:
        logger.warning("Insufficient price history — using base weights")
        base = pd.Series(
            {t: SPY_BASE_WEIGHTS_JUNE_2018.get(t, 0.0) for t in tickers}
        )
        return base / base.sum()

    # Use most recent date as rebalance date
    rebalance_date = prices.index[-1]

    weights = compute_spy_weights_at_date(
        prices=prices,
        rebalance_date=rebalance_date,
        tickers=tickers,
        lookback_days=252,
    )

    print(f"  SPY weights computed as of {rebalance_date.date()}")
    _print_weights(weights)

    return weights


def _print_weights(weights: pd.Series) -> None:
    """Print weights in a readable format."""
    print("\n  SPY Sector Weights:")
    for ticker, w in weights.sort_values(ascending=False).items():
        bar = "█" * int(w * 40)
        print(f"    {ticker:<6} {w*100:>6.2f}%  {bar}")
    print(f"    {'TOTAL':<6} {weights.sum()*100:>6.2f}%\n")
