"""
tests/test_black_litterman.py
-----------------------------
Unit tests for the Black-Litterman model.
Run with: pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest

from src.optimizer.black_litterman import (
    compute_equilibrium_returns,
    build_views,
    black_litterman,
)
from src.optimizer.covariance import compute_covariance


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_returns(n_days=500, n_assets=5, seed=42):
    np.random.seed(seed)
    tickers = [f"ETF{i}" for i in range(n_assets)]
    data    = np.random.randn(n_days, n_assets) * 0.01
    idx     = pd.bdate_range("2020-01-02", periods=n_days)
    return pd.DataFrame(data, index=idx, columns=tickers)


def make_analyst_data(tickers, implied_returns=None):
    """Create synthetic analyst data for testing."""
    if implied_returns is None:
        implied_returns = [0.10, 0.05, 0.08, 0.03, 0.12]
    rows = []
    for i, ticker in enumerate(tickers):
        rows.append({
            "implied_return": implied_returns[i],
            "analyst_count" : 5,
            "recommendation": "buy",
        })
    return pd.DataFrame(rows, index=tickers)


# ---------------------------------------------------------------------------
# Equilibrium return tests
# ---------------------------------------------------------------------------

class TestEquilibriumReturns:
    def test_equilibrium_returns_shape(self):
        returns = make_returns()
        cov     = compute_covariance(returns, method="sample")
        weights = pd.Series(
            np.ones(5) / 5,
            index=returns.columns
        )
        pi = compute_equilibrium_returns(weights, cov)
        assert len(pi) == 5

    def test_equilibrium_returns_positive_for_positive_weights(self):
        returns = make_returns()
        cov     = compute_covariance(returns, method="sample")
        weights = pd.Series(
            np.ones(5) / 5,
            index=returns.columns
        )
        pi = compute_equilibrium_returns(weights, cov, risk_aversion=2.5)
        assert (pi > 0).all()

    def test_equilibrium_returns_indexed_correctly(self):
        returns = make_returns()
        cov     = compute_covariance(returns, method="sample")
        weights = pd.Series(
            np.ones(5) / 5,
            index=returns.columns
        )
        pi = compute_equilibrium_returns(weights, cov)
        assert list(pi.index) == list(returns.columns)


# ---------------------------------------------------------------------------
# Views matrix tests
# ---------------------------------------------------------------------------

class TestBuildViews:
    def setup_method(self):
        self.tickers = [f"ETF{i}" for i in range(5)]

    def test_views_shape(self):
        analyst_data = make_analyst_data(self.tickers)
        P, Q, Omega  = build_views(analyst_data, self.tickers)
        assert P.shape[1] == 5
        assert len(Q) == P.shape[0]
        assert Omega.shape == (P.shape[0], P.shape[0])

    def test_views_returns_none_when_no_data(self):
        analyst_data = make_analyst_data(
            self.tickers,
            implied_returns=[None, None, None, None, None]
        )
        analyst_data["implied_return"] = None
        P, Q, Omega = build_views(
            analyst_data, self.tickers, min_analysts=3
        )
        assert P is None
        assert Q is None
        assert Omega is None

    def test_pick_matrix_is_binary(self):
        analyst_data = make_analyst_data(self.tickers)
        P, Q, Omega  = build_views(analyst_data, self.tickers)
        unique_vals  = np.unique(P)
        assert set(unique_vals).issubset({0.0, 1.0})

    def test_omega_is_diagonal(self):
        analyst_data = make_analyst_data(self.tickers)
        P, Q, Omega  = build_views(analyst_data, self.tickers)
        off_diagonal = Omega - np.diag(np.diag(Omega))
        assert np.allclose(off_diagonal, 0)


# ---------------------------------------------------------------------------
# Black-Litterman tests
# ---------------------------------------------------------------------------

class TestBlackLitterman:
    def setup_method(self):
        returns      = make_returns()
        self.tickers = list(returns.columns)
        self.cov     = compute_covariance(returns, method="sample")
        weights      = pd.Series(np.ones(5)/5, index=self.tickers)
        self.pi      = compute_equilibrium_returns(weights, self.cov)

    def test_bl_returns_shape(self):
        analyst_data = make_analyst_data(self.tickers)
        P, Q, Omega  = build_views(analyst_data, self.tickers)
        mu_bl        = black_litterman(self.pi, self.cov, P, Q, Omega)
        assert len(mu_bl) == 5

    def test_bl_returns_indexed_correctly(self):
        analyst_data = make_analyst_data(self.tickers)
        P, Q, Omega  = build_views(analyst_data, self.tickers)
        mu_bl        = black_litterman(self.pi, self.cov, P, Q, Omega)
        assert list(mu_bl.index) == self.tickers

    def test_bl_returns_between_equilibrium_and_views(self):
        """BL returns should be a blend — not purely equilibrium or views."""
        analyst_data = make_analyst_data(
            self.tickers,
            implied_returns=[0.20, 0.20, 0.20, 0.20, 0.20]
        )
        P, Q, Omega = build_views(analyst_data, self.tickers)
        mu_bl       = black_litterman(self.pi, self.cov, P, Q, Omega)

        # BL should be pulled toward views (0.20) from equilibrium
        # but not equal to either
        assert not np.allclose(mu_bl.values, self.pi.values)
        assert not np.allclose(mu_bl.values, 0.20)

    def test_high_confidence_views_dominate(self):
        """Very small Omega (high confidence) → BL ≈ views."""
        analyst_data = make_analyst_data(
            self.tickers,
            implied_returns=[0.30, 0.30, 0.30, 0.30, 0.30]
        )
        P, Q, Omega = build_views(
            analyst_data, self.tickers,
            confidence_scale=0.0001  # very confident
        )
        mu_bl = black_litterman(self.pi, self.cov, P, Q, Omega, tau=0.05)
        # With very high confidence, BL should be close to views
        assert np.allclose(mu_bl.values, 0.30, atol=0.05)

    def test_bl_returns_finite(self):
        """No NaN or inf values in output."""
        analyst_data = make_analyst_data(self.tickers)
        P, Q, Omega  = build_views(analyst_data, self.tickers)
        mu_bl        = black_litterman(self.pi, self.cov, P, Q, Omega)
        assert np.isfinite(mu_bl.values).all()
