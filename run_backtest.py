"""
run_backtest.py
---------------
Entry point for Phase 3 — Backtesting.
Runs the full walk-forward backtest and compares portfolio vs SPY.

Run with: python run_backtest.py
"""

import numpy as np
import pandas as pd

from src.data import build_data_pipeline
from src.optimizer import run_walk_forward
from src.backtest import (
    simulate_portfolio,
    simulate_benchmark,
    compute_all_metrics,
    compute_win_rate,
    print_metrics_table,
    plot_all,
)


def main():
    # -----------------------------------------------------------------------
    # Step 1: Load data
    # -----------------------------------------------------------------------
    print("Loading data pipeline...")
    data = build_data_pipeline(verbose=False)
    log_returns    = data["log_returns"]
    simple_returns = data["simple_returns"]
    etfs           = data["etfs"]
    benchmark      = data["benchmark"]
    config         = data["config"]

    # -----------------------------------------------------------------------
    # Step 2: Run walk-forward optimization to get rebalance weights
    # -----------------------------------------------------------------------
    print("Running walk-forward optimization...")
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

    print(f"Walk-forward complete: {len(weights_df)} rebalance periods")

    # -----------------------------------------------------------------------
    # Step 3: Simulate portfolio and benchmark NAV
    # -----------------------------------------------------------------------
    print("\nSimulating portfolio NAV...")
    portfolio_sim = simulate_portfolio(
        simple_returns=simple_returns,
        weights_df=weights_df,
        etfs=etfs,
        transaction_cost_bps=config.optimizer.transaction_cost_bps,
        initial_nav=10_000.0,
    )

    print("Simulating SPY benchmark NAV...")
    benchmark_sim = simulate_benchmark(
        simple_returns=simple_returns,
        benchmark=benchmark,
        start_date=portfolio_sim.index[0],
        initial_nav=10_000.0,
    )

    # Align on common dates
    common_dates = portfolio_sim.index.intersection(benchmark_sim.index)
    portfolio_sim = portfolio_sim.loc[common_dates]
    benchmark_sim = benchmark_sim.loc[common_dates]

    # -----------------------------------------------------------------------
    # Step 4: Compute performance metrics
    # -----------------------------------------------------------------------
    print("\nComputing performance metrics...")
    metrics_df = compute_all_metrics(
        portfolio_nav=portfolio_sim["nav"],
        portfolio_returns=portfolio_sim["daily_return"],
        benchmark_nav=benchmark_sim["nav"],
        benchmark_returns=benchmark_sim["daily_return"],
        risk_free_rate=config.performance.risk_free_rate,
    )

    win_rate = compute_win_rate(
        portfolio_daily=portfolio_sim["daily_return"],
        benchmark_daily=benchmark_sim["daily_return"],
    )

    print_metrics_table(metrics_df, win_rate)

    # -----------------------------------------------------------------------
    # Step 5: Print final NAV comparison
    # -----------------------------------------------------------------------
    port_final = portfolio_sim["nav"].iloc[-1]
    bench_final = benchmark_sim["nav"].iloc[-1]
    start_date  = portfolio_sim.index[0].date()
    end_date    = portfolio_sim.index[-1].date()

    print(f"Backtest period : {start_date} → {end_date}")
    print(f"Starting NAV    : $10,000.00")
    print(f"Portfolio final : ${port_final:,.2f}")
    print(f"SPY final       : ${bench_final:,.2f}")
    print(f"Difference      : ${port_final - bench_final:,.2f}")

    # -----------------------------------------------------------------------
    # Step 6: Generate charts
    # -----------------------------------------------------------------------
    print("\nGenerating performance charts...")
    plot_all(
        portfolio_nav=portfolio_sim["nav"],
        benchmark_nav=benchmark_sim["nav"],
        portfolio_returns=portfolio_sim["daily_return"],
        benchmark_returns=benchmark_sim["daily_return"],
        risk_free_rate=config.performance.risk_free_rate,
        charts_dir="data/charts",
    )

    print("\nPhase 3 complete. Ready for Phase 4 — Performance Analytics.")
    return portfolio_sim, benchmark_sim, metrics_df


if __name__ == "__main__":
    portfolio_sim, benchmark_sim, metrics_df = main()
