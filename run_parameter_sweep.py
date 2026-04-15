"""
run_parameter_sweep.py
----------------------
Tests multiple combinations of weight_max and rebalance_frequency
and presents a side-by-side comparison table.

Run with: python run_parameter_sweep.py
"""

import numpy as np
import pandas as pd
from itertools import product

from src.data import build_data_pipeline
from src.optimizer import run_walk_forward
from src.backtest import (
    simulate_portfolio,
    simulate_benchmark,
    compute_all_metrics,
    compute_win_rate,
)


# ---------------------------------------------------------------------------
# Parameter grid — modify these to test different combinations
# ---------------------------------------------------------------------------

WEIGHT_MAX_VALUES      = [0.30, 0.40, 0.50]        # max weight per ETF
REBALANCE_FREQUENCIES  = ["monthly", "quarterly"]   # rebalance frequency
TRANSACTION_COST_BPS   = 5.0                        # keep fixed
INITIAL_NAV            = 10_000.0


def run_single_backtest(
    log_returns,
    simple_returns,
    etfs,
    benchmark,
    config,
    weight_max,
    rebalance_frequency,
) -> dict:
    """Run a single backtest for one parameter combination."""

    # Walk-forward optimization
    weights_df = run_walk_forward(
        log_returns=log_returns,
        etfs=etfs,
        lookback_days=config.optimizer.lookback_days,
        rebalance_frequency=rebalance_frequency,
        weight_min=config.optimizer.weight_min,
        weight_max=weight_max,
        transaction_cost_bps=TRANSACTION_COST_BPS,
        risk_aversion=1.0,
        risk_free_rate=config.performance.risk_free_rate,
        cov_method="ledoit_wolf",
    )

    # Simulate portfolio
    portfolio_sim = simulate_portfolio(
        simple_returns=simple_returns,
        weights_df=weights_df,
        etfs=etfs,
        transaction_cost_bps=TRANSACTION_COST_BPS,
        initial_nav=INITIAL_NAV,
    )

    # Simulate benchmark
    benchmark_sim = simulate_benchmark(
        simple_returns=simple_returns,
        benchmark=benchmark,
        start_date=portfolio_sim.index[0],
        initial_nav=INITIAL_NAV,
    )

    # Align dates
    common = portfolio_sim.index.intersection(benchmark_sim.index)
    portfolio_sim = portfolio_sim.loc[common]
    benchmark_sim = benchmark_sim.loc[common]

    # Compute metrics
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

    # Compute average monthly turnover
    turnover = weights_df.diff().abs().sum(axis=1).dropna()

    return {
        "weight_max"           : f"{int(weight_max*100)}%",
        "rebalance_freq"       : rebalance_frequency,
        "ann_return"           : metrics_df.loc["Annualized Return (%)", "Portfolio"],
        "ann_vol"              : metrics_df.loc["Annualized Volatility (%)", "Portfolio"],
        "sharpe"               : metrics_df.loc["Sharpe Ratio", "Portfolio"],
        "max_drawdown"         : metrics_df.loc["Max Drawdown (%)", "Portfolio"],
        "avg_drawdown"         : metrics_df.loc["Avg Drawdown (%)", "Portfolio"],
        "calmar"               : metrics_df.loc["Calmar Ratio", "Portfolio"],
        "win_rate"             : round(win_rate, 1),
        "final_nav"            : round(portfolio_sim["nav"].iloc[-1], 2),
        "avg_turnover"         : round(turnover.mean() * 100, 1),
        "return_vs_spy"        : metrics_df.loc["Annualized Return (%)", "Difference"],
        "sharpe_vs_spy"        : metrics_df.loc["Sharpe Ratio", "Difference"],
    }


def print_sweep_results(results: list[dict], spy_metrics: dict) -> None:
    """Print a formatted comparison table of all parameter combinations."""

    df = pd.DataFrame(results)

    # Sort by Sharpe ratio descending
    df = df.sort_values("sharpe", ascending=False).reset_index(drop=True)
    df.index += 1  # rank starts at 1

    print("\n" + "=" * 90)
    print("PARAMETER SWEEP RESULTS — sorted by Sharpe Ratio")
    print("=" * 90)

    # Main metrics table
    display_cols = {
        "weight_max"    : "Max Weight",
        "rebalance_freq": "Frequency",
        "ann_return"    : "Ann. Return%",
        "ann_vol"       : "Volatility%",
        "sharpe"        : "Sharpe",
        "max_drawdown"  : "Max DD%",
        "calmar"        : "Calmar",
        "win_rate"      : "Win Rate%",
        "avg_turnover"  : "Avg Turnover%",
        "final_nav"     : "Final NAV $",
    }

    display_df = df[list(display_cols.keys())].rename(columns=display_cols)
    print(display_df.to_string())

    # SPY benchmark row for comparison
    print("\n" + "-" * 90)
    print(f"SPY Benchmark  | "
          f"Ann. Return: {spy_metrics['ann_return']}% | "
          f"Volatility: {spy_metrics['ann_vol']}% | "
          f"Sharpe: {spy_metrics['sharpe']} | "
          f"Max DD: {spy_metrics['max_drawdown']}% | "
          f"Final NAV: ${spy_metrics['final_nav']:,.2f}")
    print("-" * 90)

    # Vs SPY comparison
    print("\n--- Alpha vs SPY (Portfolio minus SPY) ---")
    alpha_cols = {
        "weight_max"    : "Max Weight",
        "rebalance_freq": "Frequency",
        "return_vs_spy" : "Return Alpha%",
        "sharpe_vs_spy" : "Sharpe Alpha",
        "win_rate"      : "Win Rate%",
        "avg_turnover"  : "Avg Turnover%",
    }
    alpha_df = df[list(alpha_cols.keys())].rename(columns=alpha_cols)
    print(alpha_df.to_string())
    print("=" * 90)

    # Best combination
    best = df.iloc[0]
    print(f"\nBest combination by Sharpe: "
          f"Max Weight={best['weight_max']}, "
          f"Frequency={best['rebalance_freq']}, "
          f"Sharpe={best['sharpe']}")
    print()


def main():
    # -----------------------------------------------------------------------
    # Load data once — reuse across all parameter combinations
    # -----------------------------------------------------------------------
    print("Loading data pipeline...")
    data = build_data_pipeline(verbose=False)
    log_returns    = data["log_returns"]
    simple_returns = data["simple_returns"]
    etfs           = data["etfs"]
    benchmark      = data["benchmark"]
    config         = data["config"]

    # -----------------------------------------------------------------------
    # Run SPY benchmark once
    # -----------------------------------------------------------------------
    # Use first rebalance date from a sample run as start date
    sample_weights = run_walk_forward(
        log_returns=log_returns,
        etfs=etfs,
        lookback_days=config.optimizer.lookback_days,
        rebalance_frequency="monthly",
        weight_min=config.optimizer.weight_min,
        weight_max=0.40,
        transaction_cost_bps=TRANSACTION_COST_BPS,
        risk_aversion=1.0,
        risk_free_rate=config.performance.risk_free_rate,
    )
    start_date = sample_weights.index[0]

    benchmark_sim = simulate_benchmark(
        simple_returns=simple_returns,
        benchmark=benchmark,
        start_date=start_date,
        initial_nav=INITIAL_NAV,
    )

    spy_metrics = {
        "ann_return"  : round((benchmark_sim["daily_return"].mean() * 252 * 100), 2),
        "ann_vol"     : round((benchmark_sim["daily_return"].std() * (252**0.5) * 100), 2),
        "sharpe"      : round(
            (benchmark_sim["daily_return"].mean() * 252 - config.performance.risk_free_rate) /
            (benchmark_sim["daily_return"].std() * (252**0.5)), 3
        ),
        "max_drawdown": round(
            ((benchmark_sim["nav"] - benchmark_sim["nav"].cummax()) /
             benchmark_sim["nav"].cummax()).min() * 100, 2
        ),
        "final_nav"   : round(benchmark_sim["nav"].iloc[-1], 2),
        "ann_vol"     : round(benchmark_sim["daily_return"].std() * (252**0.5) * 100, 2),
    }

    # -----------------------------------------------------------------------
    # Parameter sweep
    # -----------------------------------------------------------------------
    combinations = list(product(WEIGHT_MAX_VALUES, REBALANCE_FREQUENCIES))
    total = len(combinations)

    print(f"\nRunning {total} parameter combinations...")
    print(f"Weight max values    : {WEIGHT_MAX_VALUES}")
    print(f"Rebalance frequencies: {REBALANCE_FREQUENCIES}")
    print(f"Transaction cost     : {TRANSACTION_COST_BPS} bps (fixed)\n")

    results = []

    for i, (weight_max, freq) in enumerate(combinations, 1):
        print(f"  [{i}/{total}] weight_max={int(weight_max*100)}%, "
              f"frequency={freq} ...", end=" ")

        result = run_single_backtest(
            log_returns=log_returns,
            simple_returns=simple_returns,
            etfs=etfs,
            benchmark=benchmark,
            config=config,
            weight_max=weight_max,
            rebalance_frequency=freq,
        )
        results.append(result)
        print(f"Sharpe={result['sharpe']:.3f}, "
              f"Return={result['ann_return']}%")

    # -----------------------------------------------------------------------
    # Print results table
    # -----------------------------------------------------------------------
    print_sweep_results(results, spy_metrics)

    return pd.DataFrame(results)


if __name__ == "__main__":
    results_df = main()
