"""
tests/test_optimizer.py
-----------------------
Unit tests for the MVO engine: covariance estimation, optimization, and rebalancing.
Run with: pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest

from src.optimizer.covariance import compute_covariance, compute_expected_returns
from src.optimizer.optimizer import optimize_portfolio, compute_portfolio_metrics
from src.optimizer.rebalance import get_rebalance_dates, run_walk_forward


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_returns(n_days=500, n_assets=5, seed=42):
    """Generate synthetic daily log returns."""
    np.random.seed(seed)
    tickers = [f"ETF{i}" for i in range(n_assets)]
    data = np.random.randn(n_days, n_assets) * 0.01
    idx = pd.bdate_range("2020-01-02", periods=n_days)
    return pd.DataFrame(data, index=idx, columns=tickers)


# ---------------------------------------------------------------------------
# Covariance tests
# ---------------------------------------------------------------------------

class TestCovariance:
    def test_sample_covariance_shape(self):
        returns = make_returns()
        cov = compute_covariance(returns, method="sample")
        assert cov.shape == (5, 5)

    def test_ledoit_wolf_shape(self):
        returns = make_returns()
        cov = compute_covariance(returns, method="ledoit_wolf")
        assert cov.shape == (5, 5)

    def test_covariance_is_symmetric(self):
        returns = make_returns()
        cov = compute_covariance(returns, method="ledoit_wolf")
        np.testing.assert_array_almost_equal(cov.values, cov.values.T)

    def test_covariance_is_positive_definite(self):
        returns = make_returns()
        cov = compute_covariance(returns, method="ledoit_wolf")
        eigenvalues = np.linalg.eigvalsh(cov.values)
        assert np.all(eigenvalues > 0), "Covariance matrix must be positive definite"

    def test_expected_returns_shape(self):
        returns = make_returns()
        mu = compute_expected_returns(returns)
        assert len(mu) == 5

    def test_expected_returns_annualized(self):
        returns = make_returns()
        mu_daily = compute_expected_returns(returns, annualize=False)
        mu_annual = compute_expected_returns(returns, annualize=True)
        np.testing.assert_array_almost_equal(mu_annual.values, mu_daily.values * 252)

    def test_invalid_method_raises(self):
        returns = make_returns()
        with pytest.raises(ValueError):
            compute_covariance(returns, method="invalid")


# ---------------------------------------------------------------------------
# Optimizer tests
# ---------------------------------------------------------------------------

class TestOptimizer:
    def setup_method(self):
        returns = make_returns()
        self.mu = compute_expected_returns(returns)
        self.cov = compute_covariance(returns, method="ledoit_wolf")
        self.tickers = self.mu.index.tolist()

    def test_weights_sum_to_one(self):
        weights = optimize_portfolio(self.mu, self.cov)
        assert abs(weights.sum() - 1.0) < 1e-4

    def test_weights_non_negative(self):
        weights = optimize_portfolio(self.mu, self.cov, weight_min=0.0)
        assert (weights >= -1e-6).all()

    def test_weights_respect_max_constraint(self):
        weights = optimize_portfolio(self.mu, self.cov, weight_max=0.40)
        assert (weights <= 0.40 + 1e-4).all()

    def test_weights_indexed_by_tickers(self):
        weights = optimize_portfolio(self.mu, self.cov)
        assert list(weights.index) == self.tickers

    def test_transaction_cost_reduces_turnover(self):
        """Higher transaction costs should produce weights closer to current weights."""
        current = pd.Series(np.ones(5) / 5, index=self.tickers)
        w_low_cost = optimize_portfolio(
            self.mu, self.cov, current_weights=current, transaction_cost_bps=1
        )
        w_high_cost = optimize_portfolio(
            self.mu, self.cov, current_weights=current, transaction_cost_bps=100
        )
        turnover_low = (w_low_cost - current).abs().sum()
        turnover_high = (w_high_cost - current).abs().sum()
        assert turnover_high <= turnover_low + 1e-4

    def test_portfolio_metrics_keys(self):
        weights = optimize_portfolio(self.mu, self.cov)
        metrics = compute_portfolio_metrics(weights, self.mu, self.cov)
        assert "expected_return" in metrics
        assert "expected_volatility" in metrics
        assert "sharpe_ratio" in metrics

    def test_portfolio_volatility_positive(self):
        weights = optimize_portfolio(self.mu, self.cov)
        metrics = compute_portfolio_metrics(weights, self.mu, self.cov)
        assert metrics["expected_volatility"] > 0


# ---------------------------------------------------------------------------
# Rebalancing tests
# ---------------------------------------------------------------------------

class TestRebalancing:
    def test_monthly_rebalance_dates(self):
        idx = pd.bdate_range("2020-01-01", "2021-12-31")
        dates = get_rebalance_dates(idx, frequency="monthly")
        assert len(dates) == 24  # 24 months

    def test_quarterly_rebalance_dates(self):
        idx = pd.bdate_range("2020-01-01", "2021-12-31")
        dates = get_rebalance_dates(idx, frequency="quarterly")
        assert len(dates) == 8  # 8 quarters

    def test_invalid_frequency_raises(self):
        idx = pd.bdate_range("2020-01-01", "2021-12-31")
        with pytest.raises(ValueError):
            get_rebalance_dates(idx, frequency="weekly")

    def test_walk_forward_returns_dataframe(self):
        returns = make_returns(n_days=500, n_assets=5)
        etfs = returns.columns.tolist()
        weights_df = run_walk_forward(
            returns, etfs, lookback_days=252, rebalance_frequency="monthly"
        )
        assert isinstance(weights_df, pd.DataFrame)
        assert weights_df.shape[1] == 5

    def test_walk_forward_weights_sum_to_one(self):
        returns = make_returns(n_days=500, n_assets=5)
        etfs = returns.columns.tolist()
        weights_df = run_walk_forward(
            returns, etfs, lookback_days=252, rebalance_frequency="monthly"
        )
        row_sums = weights_df.sum(axis=1)
        assert (abs(row_sums - 1.0) < 1e-3).all()
