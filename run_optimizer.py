"""
run_optimizer.py
----------------
Entry point to test the Phase 2 MVO engine end-to-end.
Run with: python run_optimizer.py
"""

import pandas as pd
import numpy as np

from src.data import build_data_pipeline
from src.optimizer import (
    compute_covariance,
    compute_expected_returns,
    covariance_diagnostics,
    optimize_portfolio,
    compute_portfolio_metrics,
    print_weights,
    run_walk_forward,
)


def main():
    # -----------------------------------------------------------------------
    # Step 1: Load data
    # -----------------------------------------------------------------------
    print("Loading data pipeline...")
    data = build_data_pipeline(verbose=False)
    log_returns = data["log_returns"]
    etfs = data["etfs"]
    config = data["config"]

    # -----------------------------------------------------------------------
    # Step 2: Single-period optimization (sanity check)
    # Use full history to compute one set of optimal weights
    # -----------------------------------------------------------------------
    print("\n=== Single-Period Optimization (full history) ===")

    etf_returns = log_returns[etfs]
    mu = compute_expected_returns(etf_returns, annualize=True)
    cov = compute_covariance(etf_returns, method="ledoit_wolf", annualize=True)

    # Check covariance matrix health
    covariance_diagnostics(cov)
    
    # Eigenvalue breakdown
    eigenvalues = np.linalg.eigvalsh(cov.values)[::-1]  # sort largest first
    total = eigenvalues.sum()
    print("\n--- Eigenvalue Breakdown ---")
    for i, e in enumerate(eigenvalues):
        print(f"  Factor {i+1:>2}: {e:.6f}  ({e/total*100:.1f}% of total variance)")
    
    # Eigenvector breakdown — shows which ETFs drive each factor
    eigenvectors = np.linalg.eigh(cov.values)[1][:, ::-1]
    print("\n--- Top 3 Factor Compositions ---")
    for i in range(3):
        print(f"\nFactor {i+1} ({eigenvalues[i]/total*100:.1f}% of variance):")
        factor = pd.Series(eigenvectors[:, i], index=etfs).sort_values(key=abs, ascending=False)
        for ticker, loading in factor.items():
            bar = "█" * int(abs(loading) * 20)
            print(f"  {ticker:<6} {loading:>+.3f}  {bar}")

    # Optimize
    weights = optimize_portfolio(
        mu=mu,
        cov=cov,
        current_weights=None,
        weight_min=config.optimizer.weight_min,
        weight_max=config.optimizer.weight_max,
        transaction_cost_bps=config.optimizer.transaction_cost_bps,
        risk_aversion=1.0,
    )

    # Display weights
    print_weights(weights, title="Optimal Portfolio Weights (full history)")

    # Display metrics
    metrics = compute_portfolio_metrics(
        weights, mu, cov, risk_free_rate=config.performance.risk_free_rate
    )
    print("--- Expected Portfolio Metrics ---")
    print(f"  Expected Return     : {metrics['expected_return']:.2f}%")
    print(f"  Expected Volatility : {metrics['expected_volatility']:.2f}%")
    print(f"  Sharpe Ratio        : {metrics['sharpe_ratio']:.3f}")
    print("----------------------------------\n")

    # -----------------------------------------------------------------------
    # Step 3: Walk-forward rebalancing
    # -----------------------------------------------------------------------
    print("=== Walk-Forward Rebalancing ===")

    weights_df = run_walk_forward(
        log_returns=log_returns,
        etfs=etfs,
        lookback_days=config.optimizer.lookback_days,
        rebalance_frequency=config.optimizer.rebalance_frequency,
        weight_min=config.optimizer.weight_min,
        weight_max=config.optimizer.weight_max,
        transaction_cost_bps=config.optimizer.transaction_cost_bps,
        risk_aversion=1.0,
        risk_free_rate=config.performance.risk_free_rate,
        cov_method="ledoit_wolf",
    )

    print(f"\nRebalancing complete: {len(weights_df)} periods")
    print(f"\nFirst rebalance weights:\n{weights_df.iloc[0].round(4)}")
    print(f"\nLast rebalance weights:\n{weights_df.iloc[-1].round(4)}")
    print(f"\nAverage weights over time:\n{weights_df.mean().round(4)}")

    # Show how much the portfolio changed over time
    turnover = weights_df.diff().abs().sum(axis=1).dropna()
    print(f"\nAverage monthly turnover: {turnover.mean()*100:.2f}%")
    print(f"Max monthly turnover    : {turnover.max()*100:.2f}%")

    print("\nPhase 2 complete. Ready for Phase 3 — Backtesting.")
    return weights_df


if __name__ == "__main__":
    weights_df = main()
