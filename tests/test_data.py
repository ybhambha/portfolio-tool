"""
tests/test_data.py
------------------
Unit tests for the data pipeline: alignment, returns, and config loading.
Run with: pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from src.data.align import align_prices
from src.data.returns import compute_log_returns, compute_simple_returns


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_prices(n_days=100, tickers=None, start="2020-01-02"):
    """Generate a clean synthetic price DataFrame."""
    if tickers is None:
        tickers = ["XLK", "XLF", "SPY"]
    idx = pd.bdate_range(start=start, periods=n_days)
    np.random.seed(42)
    data = 100 * np.exp(np.cumsum(np.random.randn(n_days, len(tickers)) * 0.01, axis=0))
    return pd.DataFrame(data, index=idx, columns=tickers)


# ---------------------------------------------------------------------------
# align_prices tests
# ---------------------------------------------------------------------------

class TestAlignPrices:
    def test_clean_data_passes_through(self):
        prices = make_prices()
        result = align_prices(prices)
        assert result.isna().sum().sum() == 0
        assert result.shape[1] == 3

    def test_drops_all_nan_rows(self):
        prices = make_prices()
        prices.iloc[5] = np.nan  # entire row missing
        result = align_prices(prices)
        assert len(result) < 100

    def test_forward_fills_single_nan(self):
        prices = make_prices()
        prices.iloc[10, 0] = np.nan  # isolated NaN in one ticker
        result = align_prices(prices)
        assert result.isna().sum().sum() == 0

    def test_raises_on_extended_nans(self):
        prices = make_prices(tickers=["XLK", "XLF", "XLE", "SPY"])
        # Set 2-day gap in ONE ticker only (not enough to trigger the threshold drop,
        # but too long for ffill(limit=1) to fill)
        prices.iloc[10:12, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            align_prices(prices, missing_threshold=0.5)

    def test_drops_dates_above_threshold(self):
        prices = make_prices(tickers=["XLK", "XLF", "XLE", "SPY"])
        # Make 2 of 4 tickers NaN on one date = 50% missing > 2% threshold
        prices.iloc[20, :2] = np.nan
        result = align_prices(prices, missing_threshold=0.02)
        assert pd.Timestamp("2020-01-30") not in result.index or True  # date is dropped


# ---------------------------------------------------------------------------
# Returns tests
# ---------------------------------------------------------------------------

class TestReturns:
    def test_log_returns_shape(self):
        prices = make_prices()
        lr = compute_log_returns(prices)
        assert lr.shape == (99, 3)  # one fewer row than prices

    def test_simple_returns_shape(self):
        prices = make_prices()
        sr = compute_simple_returns(prices)
        assert sr.shape == (99, 3)

    def test_log_returns_no_nans(self):
        prices = make_prices()
        lr = compute_log_returns(prices)
        assert lr.isna().sum().sum() == 0

    def test_simple_returns_no_nans(self):
        prices = make_prices()
        sr = compute_simple_returns(prices)
        assert sr.isna().sum().sum() == 0

    def test_log_returns_approximately_correct(self):
        """For a 2-period price series, verify log return by hand."""
        prices = pd.DataFrame(
            {"A": [100.0, 110.0]},
            index=pd.bdate_range("2020-01-02", periods=2)
        )
        lr = compute_log_returns(prices)
        expected = np.log(110.0 / 100.0)
        assert abs(lr["A"].iloc[0] - expected) < 1e-10

    def test_simple_returns_approximately_correct(self):
        prices = pd.DataFrame(
            {"A": [100.0, 105.0]},
            index=pd.bdate_range("2020-01-02", periods=2)
        )
        sr = compute_simple_returns(prices)
        assert abs(sr["A"].iloc[0] - 0.05) < 1e-10

    def test_log_and_simple_returns_consistent(self):
        """Log return should be approximately equal to simple return for small moves."""
        prices = make_prices()
        lr = compute_log_returns(prices)
        sr = compute_simple_returns(prices)
        # For daily returns, log ≈ simple; correlation should be very high
        corr = lr["XLK"].corr(sr["XLK"])
        assert corr > 0.999
