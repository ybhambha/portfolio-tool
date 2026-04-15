"""
run_report.py
-------------
Entry point for Phase 4 — Performance Reporting.
Runs the full pipeline and generates an HTML performance report.

Run with: python run_report.py
"""

import os
from src.data import build_data_pipeline
from src.optimizer import run_walk_forward
from src.backtest import simulate_portfolio, simulate_benchmark
from src.reporting import generate_html_report


def main():
    # -----------------------------------------------------------------------
    # Step 1: Load data
    # -----------------------------------------------------------------------
    print("Loading data pipeline...")
    data           = build_data_pipeline(verbose=False)
    log_returns    = data["log_returns"]
    simple_returns = data["simple_returns"]
    etfs           = data["etfs"]
    benchmark      = data["benchmark"]
    config         = data["config"]

    # -----------------------------------------------------------------------
    # Step 2: Walk-forward optimization
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

    # -----------------------------------------------------------------------
    # Step 3: Simulate NAVs
    # -----------------------------------------------------------------------
    print("Simulating portfolio and benchmark NAVs...")
    portfolio_sim = simulate_portfolio(
        simple_returns=simple_returns,
        weights_df=weights_df,
        etfs=etfs,
        transaction_cost_bps=config.optimizer.transaction_cost_bps,
        initial_nav=10_000.0,
    )

    benchmark_sim = simulate_benchmark(
        simple_returns=simple_returns,
        benchmark=benchmark,
        start_date=portfolio_sim.index[0],
        initial_nav=10_000.0,
    )

    # Align on common dates
    common = portfolio_sim.index.intersection(benchmark_sim.index)
    portfolio_sim = portfolio_sim.loc[common]
    benchmark_sim = benchmark_sim.loc[common]

    # -----------------------------------------------------------------------
    # Step 4: Generate HTML report
    # -----------------------------------------------------------------------
    print("Generating HTML performance report...")
    output_path = "data/reports/performance_report.html"

    report_path = generate_html_report(
        portfolio_nav=portfolio_sim["nav"],
        benchmark_nav=benchmark_sim["nav"],
        portfolio_returns=portfolio_sim["daily_return"],
        benchmark_returns=benchmark_sim["daily_return"],
        weights_df=weights_df,
        config=config,
        output_path=output_path,
    )

    print(f"\nReport generated successfully!")
    print(f"Open this file in your browser:")
    print(f"  {os.path.abspath(report_path)}")
    print("\nPhase 4 complete. Ready for Phase 5 — Trade Signals & Email Alerts.")


if __name__ == "__main__":
    main()
