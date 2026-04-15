"""
tests/test_backtest.py
----------------------
Unit tests for the backtesting engine.
Run with: pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest.simulator import simulate_portfolio, simulate_benchmark
from src.backtest.metrics import (
    compute_cagr,
    compute_volatility,
    compute_sharpe,
    compute_drawdown_series,
    compute_max_drawdown,
    compute_win_rate,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_returns(n_days=500, tickers=None, seed=42):
    np.random.seed(seed)
    if tickers is None:
        tickers = ["XLK", "XLF", "XLE", "SPY"]
    idx = pd.bdate_range("2020-01-02", periods=n_days)
    data = np.random.randn(n_days, len(tickers)) * 0.01
    return pd.DataFrame(data, index=idx, columns=tickers)


def make_weights(n_periods=20, etfs=None):
    np.random.seed(42)
    if etfs is None:
        etfs = ["XLK", "XLF", "XLE"]
    idx = pd.bdate_range("2021-01-29", periods=n_periods, freq="BME")
    weights = np.random.dirichlet(np.ones(len(etfs)), size=n_periods)
    return pd.DataFrame(weights, index=idx, columns=etfs)


# ---------------------------------------------------------------------------
# Simulator tests
# ---------------------------------------------------------------------------

class TestSimulator:
    def test_portfolio_nav_starts_at_initial_value(self):
        returns = make_returns()
        weights = make_weights()
        result = simulate_portfolio(returns, weights, ["XLK", "XLF", "XLE"])
        assert result["nav"].iloc[0] > 0

    def test_portfolio_nav_always_positive(self):
        returns = make_returns()
        weights = make_weights()
        result = simulate_portfolio(returns, weights, ["XLK", "XLF", "XLE"])
        assert (result["nav"] > 0).all()

    def test_benchmark_nav_starts_at_initial_value(self):
        returns = make_returns()
        start = returns.index[0]
        result = simulate_benchmark(returns, "SPY", start_date=start)
        assert abs(result["nav"].iloc[0] - 10000 * (1 + returns["SPY"].iloc[0])) < 1

    def test_benchmark_nav_always_positive(self):
        returns = make_returns()
        start = returns.index[0]
        result = simulate_benchmark(returns, "SPY", start_date=start)
        assert (result["nav"] > 0).all()

    def test_portfolio_returns_same_length_as_nav(self):
        returns = make_returns()
        weights = make_weights()
        result = simulate_portfolio(returns, weights, ["XLK", "XLF", "XLE"])
        assert len(result["nav"]) == len(result["daily_return"])


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_cagr_positive_growth(self):
        nav = pd.Series([10000, 11000], index=pd.bdate_range("2020-01-02", periods=2))
        # Over very short period CAGR should be very high
        assert compute_cagr(nav) > 0

    def test_cagr_flat_nav(self):
        nav = pd.Series(
            [10000] * 252,
            index=pd.bdate_range("2020-01-02", periods=252)
        )
        assert abs(compute_cagr(nav)) < 0.001

    def test_volatility_positive(self):
        returns = make_returns()
        vol = compute_volatility(returns["SPY"])
        assert vol > 0

    def test_volatility_zero_for_constant_returns(self):
        returns = pd.Series(
            [0.001] * 252,
            index=pd.bdate_range("2020-01-02", periods=252)
        )
        assert compute_volatility(returns) < 1e-10

    def test_drawdown_series_always_non_positive(self):
        nav = pd.Series(
            [10000, 11000, 9000, 10500],
            index=pd.bdate_range("2020-01-02", periods=4)
        )
        dd = compute_drawdown_series(nav)
        assert (dd <= 0).all()

    def test_max_drawdown_negative(self):
        nav = pd.Series(
            [10000, 11000, 8000, 10000],
            index=pd.bdate_range("2020-01-02", periods=4)
        )
        max_dd = compute_max_drawdown(nav)
        assert max_dd < 0

    def test_max_drawdown_correct_value(self):
        nav = pd.Series(
            [10000, 11000, 8000, 10000],
            index=pd.bdate_range("2020-01-02", periods=4)
        )
        max_dd = compute_max_drawdown(nav)
        # Peak = 11000, trough = 8000 → drawdown = (8000-11000)/11000
        expected = (8000 - 11000) / 11000
        assert abs(max_dd - expected) < 1e-6

    def test_sharpe_positive_for_good_strategy(self):
        # Strategy with consistent positive returns should have positive Sharpe
        returns = pd.Series(
            [0.001] * 252,
            index=pd.bdate_range("2020-01-02", periods=252)
        )
        sharpe = compute_sharpe(returns, risk_free_rate=0.0)
        assert sharpe > 0

    def test_win_rate_between_0_and_100(self):
        returns = make_returns()
        win_rate = compute_win_rate(returns["XLK"], returns["SPY"])
        assert 0 <= win_rate <= 100
