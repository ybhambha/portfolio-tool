"""
src/data/__init__.py
--------------------
Single entry point for the full data pipeline.
Call build_data_pipeline() to get prices, log returns, and simple returns
ready for the optimizer and backtester.
"""

import logging
from src.config import load_config
from src.data.fetch import fetch_prices
from src.data.align import align_prices, log_data_summary
from src.data.returns import compute_log_returns, compute_simple_returns, returns_summary

logger = logging.getLogger(__name__)


def build_data_pipeline(config_path: str = "config.yaml", verbose: bool = True) -> dict:
    """
    Run the full data pipeline:
      1. Load config
      2. Fetch prices (from cache or yfinance)
      3. Align and clean
      4. Compute log and simple returns

    Parameters
    ----------
    config_path : path to config.yaml
    verbose     : if True, print data summary and return stats

    Returns
    -------
    dict with keys:
      "prices"         : pd.DataFrame, aligned adjusted closing prices
      "log_returns"    : pd.DataFrame, daily log returns
      "simple_returns" : pd.DataFrame, daily simple returns
      "etfs"           : list of ETF ticker strings
      "benchmark"      : benchmark ticker string (SPY)
      "config"         : AppConfig object
    """
    cfg = load_config(config_path)
    logger.info(f"Building data pipeline | backend={cfg.storage.backend} | "
                f"universe={cfg.universe.etfs} | benchmark={cfg.universe.benchmark}")

    # Step 1: Fetch
    prices_raw = fetch_prices(
        tickers=cfg.all_tickers,
        start=cfg.data.start_date,
        end=cfg.data.end_date,
        storage_cfg=cfg.storage.model_dump(),
        price_col=cfg.data.price_col,
    )

    # Step 2: Align
    prices = align_prices(prices_raw, missing_threshold=cfg.data.missing_threshold)

    if verbose:
        log_data_summary(prices)

    # Step 3: Returns
    log_returns = compute_log_returns(prices)
    simple_returns = compute_simple_returns(prices)

    if verbose:
        returns_summary(log_returns)

    return {
        "prices": prices,
        "log_returns": log_returns,
        "simple_returns": simple_returns,
        "etfs": cfg.universe.etfs,
        "benchmark": cfg.universe.benchmark,
        "config": cfg,
    }
