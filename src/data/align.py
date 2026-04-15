"""
align.py
--------
Aligns price DataFrames across all tickers:
  - Drops dates where too many assets have missing data
  - Forward-fills isolated NaNs (e.g. trading halts, one-day gaps)
  - Raises on unresolvable gaps so problems are never silently swallowed
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def align_prices(prices: pd.DataFrame, missing_threshold: float = 0.02) -> pd.DataFrame:
    """
    Clean and align a raw price DataFrame.

    Steps
    -----
    1. Drop rows where ALL tickers are NaN (weekends can slip through yfinance)
    2. Drop dates where the fraction of missing assets exceeds missing_threshold
    3. Forward-fill remaining isolated NaNs (limit=1 to avoid masking real gaps)
    4. Raise if any NaNs remain — forces the caller to deal with real data problems

    Parameters
    ----------
    prices            : raw DataFrame from fetch_prices(), shape (days, tickers)
    missing_threshold : max fraction of tickers allowed to be NaN on a given date

    Returns
    -------
    pd.DataFrame : cleaned prices, same columns, DatetimeIndex
    """

    original_shape = prices.shape

    # Step 1: drop completely empty rows
    prices = prices.dropna(how="all")

    # Step 2: drop dates where too many assets are missing
    missing_frac = prices.isna().mean(axis=1)
    bad_dates = missing_frac[missing_frac > missing_threshold].index

    if len(bad_dates) > 0:
        logger.warning(
            f"Dropping {len(bad_dates)} date(s) with >{missing_threshold * 100:.0f}% "
            f"missing assets. First few: {list(bad_dates[:5])}"
        )
        prices = prices.drop(index=bad_dates)

    # Step 3: forward-fill isolated NaNs (max 1 day gap)
    prices = prices.ffill(limit=1)

    # Step 4: fail loudly if NaNs remain
    remaining_nans = prices.isna().sum()
    total_remaining = remaining_nans.sum()

    if total_remaining > 0:
        problem_tickers = remaining_nans[remaining_nans > 0]
        raise ValueError(
            f"{total_remaining} NaN(s) remain after alignment. "
            f"These tickers likely have a shorter history than your start_date:\n"
            f"{problem_tickers.to_string()}\n\n"
            f"Fix: adjust start_date in config.yaml (XLRE starts 2015-10-08), "
            f"or drop the problematic ticker from the universe."
        )

    logger.info(
        f"align_prices: {original_shape} → {prices.shape} "
        f"({original_shape[0] - prices.shape[0]} dates dropped)"
    )

    return prices


def log_data_summary(prices: pd.DataFrame) -> None:
    """Print a quick sanity-check summary of the aligned price data."""
    print("\n--- Data Summary ---")
    print(f"Date range : {prices.index.min().date()} → {prices.index.max().date()}")
    print(f"Trading days: {len(prices)}")
    print(f"Tickers    : {list(prices.columns)}")
    print(f"NaNs       : {prices.isna().sum().sum()}")
    print(f"\nFirst prices:\n{prices.head(3).to_string()}")
    print(f"\nLast prices:\n{prices.tail(3).to_string()}")
    print("--------------------\n")
