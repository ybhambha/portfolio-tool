"""
run_bl_optimizer.py
-------------------
Runs a full backtest comparison between:

  Strategy 1: MVO with historical returns  (original approach)
  Strategy 2: MVO with BL equilibrium returns (new approach)

At each monthly rebalance date Strategy 2 computes time-varying
SPY sector weights using a rolling 252-day price-relative approach,
derives equilibrium returns, and feeds those into the MVO optimizer.

Run with: python run_bl_optimizer.py
"""

import pandas as pd
import numpy as np

from src.data import build_data_pipeline
from src.optimizer import run_walk_forward
from src.optimizer.rebalance_bl import run_walk_forward_bl
from src.optimizer.spy_weights import (
    compute_all_spy_weights,
    print_spy_weight_evolution,
)
from src.backtest import (
    simulate_portfolio,
    simulate_benchmark,
    compute_all_metrics,
    compute_win_rate,
    print_metrics_table,
)
from src.optimizer.rebalance import get_rebalance_dates


def main():
    # -----------------------------------------------------------------------
    # Step 1: Load data
    # -----------------------------------------------------------------------
    print("Loading data pipeline...")
    data           = build_data_pipeline(verbose=False)
    log_returns    = data["log_returns"]
    simple_returns = data["simple_returns"]
    prices         = data["prices"]
    etfs           = data["etfs"]
    benchmark      = data["benchmark"]
    config         = data["config"]

    # -----------------------------------------------------------------------
    # Step 2: Show how SPY weights evolved over the backtest period
    # -----------------------------------------------------------------------
    print("\nComputing time-varying SPY sector weights...")
    rebalance_dates = get_rebalance_dates(
        log_returns[etfs].index,
        frequency=config.optimizer.rebalance_frequency,
    )
    first_valid = log_returns.index[config.optimizer.lookback_days]
    rebalance_dates = [d for d in rebalance_dates if d >= first_valid]

    spy_weights_history = compute_all_spy_weights(
        prices=prices[etfs],
        rebalance_dates=rebalance_dates,
        tickers=etfs,
        lookback_days=config.optimizer.lookback_days,
    )

    print_spy_weight_evolution(
        spy_weights_history,
        tickers=["XLK", "XLF", "XLE", "XLV", "XLC"],
        sample_dates=6,
    )

    # -----------------------------------------------------------------------
    # Step 3: Strategy 1 — Historical MVO (original)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STRATEGY 1: MVO with Historical Returns")
    print("=" * 60)

    weights_historical = run_walk_forward(
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

    port_hist = simulate_portfolio(
        simple_returns=simple_returns,
        weights_df=weights_historical,
        etfs=etfs,
        transaction_cost_bps=config.optimizer.transaction_cost_bps,
        initial_nav=10_000.0,
    )

    # -----------------------------------------------------------------------
    # Step 4: Strategy 2 — BL with time-varying SPY weights
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STRATEGY 2: MVO with Black-Litterman Equilibrium Returns")
    print("=" * 60)

    weights_bl = run_walk_forward_bl(
        log_returns=log_returns,
        prices=prices,
        etfs=etfs,
        lookback_days=config.optimizer.lookback_days,
        rebalance_frequency=config.optimizer.rebalance_frequency,
        weight_min=config.optimizer.weight_min,
        weight_max=config.optimizer.weight_max,
        transaction_cost_bps=config.optimizer.transaction_cost_bps,
        risk_aversion=2.5,
        risk_free_rate=config.performance.risk_free_rate,
        cov_method="ledoit_wolf",
        tau=0.05,
    )

    port_bl = simulate_portfolio(
        simple_returns=simple_returns,
        weights_df=weights_bl,
        etfs=etfs,
        transaction_cost_bps=config.optimizer.transaction_cost_bps,
        initial_nav=10_000.0,
    )

    # -----------------------------------------------------------------------
    # Step 5: Benchmark — SPY buy and hold
    # -----------------------------------------------------------------------
    start_date    = max(port_hist.index[0], port_bl.index[0])
    benchmark_sim = simulate_benchmark(
        simple_returns=simple_returns,
        benchmark=benchmark,
        start_date=start_date,
        initial_nav=10_000.0,
    )

    # Align all on common dates
    common = (
        port_hist.index
        .intersection(port_bl.index)
        .intersection(benchmark_sim.index)
    )
    port_hist     = port_hist.loc[common]
    port_bl       = port_bl.loc[common]
    benchmark_sim = benchmark_sim.loc[common]

    # -----------------------------------------------------------------------
    # Step 6: Compare results
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STRATEGY COMPARISON")
    print("=" * 60)

    # Historical MVO metrics
    metrics_hist = compute_all_metrics(
        portfolio_nav=port_hist["nav"],
        portfolio_returns=port_hist["daily_return"],
        benchmark_nav=benchmark_sim["nav"],
        benchmark_returns=benchmark_sim["daily_return"],
        risk_free_rate=config.performance.risk_free_rate,
    )
    win_rate_hist = compute_win_rate(
        port_hist["daily_return"],
        benchmark_sim["daily_return"],
    )

    # BL metrics
    metrics_bl = compute_all_metrics(
        portfolio_nav=port_bl["nav"],
        portfolio_returns=port_bl["daily_return"],
        benchmark_nav=benchmark_sim["nav"],
        benchmark_returns=benchmark_sim["daily_return"],
        risk_free_rate=config.performance.risk_free_rate,
    )
    win_rate_bl = compute_win_rate(
        port_bl["daily_return"],
        benchmark_sim["daily_return"],
    )

    # Print side-by-side comparison
    print(f"\n{'Metric':<28} {'Hist MVO':>12} {'BL MVO':>12} {'SPY':>12}")
    print("-" * 68)

    metrics_to_show = [
        "Annualized Return (%)",
        "Annualized Volatility (%)",
        "Sharpe Ratio",
        "Max Drawdown (%)",
        "Avg Drawdown (%)",
        "Calmar Ratio",
    ]

    for metric in metrics_to_show:
        hist_val = metrics_hist.loc[metric, "Portfolio"]
        bl_val   = metrics_bl.loc[metric, "Portfolio"]
        spy_val  = metrics_hist.loc[metric, "SPY"]
        print(
            f"{metric:<28} "
            f"{hist_val:>12} "
            f"{bl_val:>12} "
            f"{spy_val:>12}"
        )

    print("-" * 68)
    print(
        f"{'Monthly Win Rate vs SPY':<28} "
        f"{win_rate_hist:>11.1f}% "
        f"{win_rate_bl:>11.1f}% "
        f"{'—':>12}"
    )
    print(
        f"{'Final NAV ($10k start)':<28} "
        f"${port_hist['nav'].iloc[-1]:>10,.2f} "
        f"${port_bl['nav'].iloc[-1]:>10,.2f} "
        f"${benchmark_sim['nav'].iloc[-1]:>10,.2f}"
    )
    print("=" * 68)

    # -----------------------------------------------------------------------
    # Step 7: Show average weights for each strategy
    # -----------------------------------------------------------------------
    print("\n--- Average Weights Comparison ---")
    print(f"{'ETF':<6} {'Hist MVO':>10} {'BL MVO':>10} {'Difference':>12}")
    print("-" * 42)

    avg_hist = weights_historical.mean().sort_values(ascending=False)
    avg_bl   = weights_bl.mean()

    for ticker in avg_hist.index:
        h    = avg_hist[ticker]
        b    = avg_bl.get(ticker, 0)
        diff = b - h
        sign = "+" if diff > 0 else ""
        print(
            f"{ticker:<6} "
            f"{h*100:>9.2f}% "
            f"{b*100:>9.2f}% "
            f"{sign}{diff*100:>10.2f}%"
        )

    print("\nDone. Use these results to decide which strategy to trade.")
    return weights_historical, weights_bl, metrics_hist, metrics_bl


if __name__ == "__main__":
    weights_historical, weights_bl, metrics_hist, metrics_bl = main()
