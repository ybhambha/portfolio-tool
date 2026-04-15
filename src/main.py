"""
src/main.py
-----------
Main CLI entry point for the portfolio construction tool.
Uses Typer to provide clean command-line interface.

Available commands:
  python -m src.main run-data        → Phase 1: fetch and cache data
  python -m src.main run-backtest    → Phase 3: run full backtest
  python -m src.main run-report      → Phase 4: generate HTML report
  python -m src.main run-signal      → Phase 5: check today's signal
  python -m src.main run-scheduler   → Phase 5: start daily daemon
  python -m src.main run-sweep       → parameter sweep
"""

import typer
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = typer.Typer(
    name="portfolio-tool",
    help="MVO Sector ETF Portfolio Construction Tool",
    add_completion=False,
)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command()
def run_data(
    config: str = typer.Option("config.yaml", help="Path to config.yaml"),
    verbose: bool = typer.Option(True, help="Print data summary"),
):
    """Phase 1: Fetch and cache price data."""
    from src.data import build_data_pipeline
    print("Running data pipeline...")
    data = build_data_pipeline(config_path=config, verbose=verbose)
    print(f"\nData pipeline complete.")
    print(f"  Tickers      : {data['etfs'] + [data['benchmark']]}")
    print(f"  Trading days : {len(data['prices'])}")
    print(f"  Date range   : {data['prices'].index[0].date()} → "
          f"{data['prices'].index[-1].date()}")


@app.command()
def run_backtest(
    config: str = typer.Option("config.yaml", help="Path to config.yaml"),
    initial_nav: float = typer.Option(10_000.0, help="Starting NAV in dollars"),
):
    """Phase 3: Run full walk-forward backtest and print results."""
    from src.data import build_data_pipeline
    from src.optimizer import run_walk_forward
    from src.backtest import (
        simulate_portfolio, simulate_benchmark,
        compute_all_metrics, compute_win_rate,
        print_metrics_table, plot_all,
    )

    print("Loading data...")
    data           = build_data_pipeline(config_path=config, verbose=False)
    log_returns    = data["log_returns"]
    simple_returns = data["simple_returns"]
    etfs           = data["etfs"]
    benchmark      = data["benchmark"]
    cfg            = data["config"]

    print("Running walk-forward optimization...")
    weights_df = run_walk_forward(
        log_returns=log_returns,
        etfs=etfs,
        lookback_days=cfg.optimizer.lookback_days,
        rebalance_frequency=cfg.optimizer.rebalance_frequency,
        weight_min=cfg.optimizer.weight_min,
        weight_max=cfg.optimizer.weight_max,
        transaction_cost_bps=cfg.optimizer.transaction_cost_bps,
        risk_aversion=1.0,
        risk_free_rate=cfg.performance.risk_free_rate,
    )

    portfolio_sim = simulate_portfolio(
        simple_returns=simple_returns,
        weights_df=weights_df,
        etfs=etfs,
        transaction_cost_bps=cfg.optimizer.transaction_cost_bps,
        initial_nav=initial_nav,
    )

    benchmark_sim = simulate_benchmark(
        simple_returns=simple_returns,
        benchmark=benchmark,
        start_date=portfolio_sim.index[0],
        initial_nav=initial_nav,
    )

    common = portfolio_sim.index.intersection(benchmark_sim.index)
    portfolio_sim = portfolio_sim.loc[common]
    benchmark_sim = benchmark_sim.loc[common]

    metrics_df = compute_all_metrics(
        portfolio_nav=portfolio_sim["nav"],
        portfolio_returns=portfolio_sim["daily_return"],
        benchmark_nav=benchmark_sim["nav"],
        benchmark_returns=benchmark_sim["daily_return"],
        risk_free_rate=cfg.performance.risk_free_rate,
    )

    win_rate = compute_win_rate(
        portfolio_sim["daily_return"],
        benchmark_sim["daily_return"],
    )

    print_metrics_table(metrics_df, win_rate)
    plot_all(
        portfolio_nav=portfolio_sim["nav"],
        benchmark_nav=benchmark_sim["nav"],
        portfolio_returns=portfolio_sim["daily_return"],
        benchmark_returns=benchmark_sim["daily_return"],
        risk_free_rate=cfg.performance.risk_free_rate,
    )


@app.command()
def run_report(
    config: str = typer.Option("config.yaml", help="Path to config.yaml"),
    output: str = typer.Option(
        "data/reports/performance_report.html",
        help="Output path for HTML report"
    ),
    initial_nav: float = typer.Option(10_000.0, help="Starting NAV in dollars"),
):
    """Phase 4: Generate HTML performance report."""
    import os
    from src.data import build_data_pipeline
    from src.optimizer import run_walk_forward
    from src.backtest import simulate_portfolio, simulate_benchmark
    from src.reporting import generate_html_report

    print("Loading data...")
    data           = build_data_pipeline(config_path=config, verbose=False)
    log_returns    = data["log_returns"]
    simple_returns = data["simple_returns"]
    etfs           = data["etfs"]
    benchmark      = data["benchmark"]
    cfg            = data["config"]

    print("Running walk-forward optimization...")
    weights_df = run_walk_forward(
        log_returns=log_returns,
        etfs=etfs,
        lookback_days=cfg.optimizer.lookback_days,
        rebalance_frequency=cfg.optimizer.rebalance_frequency,
        weight_min=cfg.optimizer.weight_min,
        weight_max=cfg.optimizer.weight_max,
        transaction_cost_bps=cfg.optimizer.transaction_cost_bps,
        risk_aversion=1.0,
        risk_free_rate=cfg.performance.risk_free_rate,
    )

    portfolio_sim = simulate_portfolio(
        simple_returns=simple_returns,
        weights_df=weights_df,
        etfs=etfs,
        transaction_cost_bps=cfg.optimizer.transaction_cost_bps,
        initial_nav=initial_nav,
    )

    benchmark_sim = simulate_benchmark(
        simple_returns=simple_returns,
        benchmark=benchmark,
        start_date=portfolio_sim.index[0],
        initial_nav=initial_nav,
    )

    common = portfolio_sim.index.intersection(benchmark_sim.index)
    portfolio_sim = portfolio_sim.loc[common]
    benchmark_sim = benchmark_sim.loc[common]

    print("Generating HTML report...")
    report_path = generate_html_report(
        portfolio_nav=portfolio_sim["nav"],
        benchmark_nav=benchmark_sim["nav"],
        portfolio_returns=portfolio_sim["daily_return"],
        benchmark_returns=benchmark_sim["daily_return"],
        weights_df=weights_df,
        config=cfg,
        output_path=output,
    )

    print(f"\nReport saved to: {os.path.abspath(report_path)}")


@app.command()
def run_signal(
    config: str  = typer.Option("config.yaml", help="Path to config.yaml"),
    value: float = typer.Option(10_000.0, help="Portfolio value in dollars"),
    email: bool  = typer.Option(False, help="Send email alert if rebalance day"),
):
    """Phase 5: Check today's rebalance signal and print trade list."""
    from src.config import load_config
    from src.alerts.signals import generate_rebalance_signal
    from src.alerts.email_alert import send_rebalance_email, print_trade_list

    cfg    = load_config(config)
    signal = generate_rebalance_signal(config=cfg, portfolio_value=value)
    print_trade_list(signal)

    if email and signal["is_rebalance"]:
        send_rebalance_email(signal)


@app.command()
def run_scheduler(
    config: str  = typer.Option("config.yaml", help="Path to config.yaml"),
    value: float = typer.Option(10_000.0, help="Portfolio value in dollars"),
    hour: int    = typer.Option(16, help="Hour to run daily check (ET, 24h)"),
    minute: int  = typer.Option(30, help="Minute to run daily check"),
):
    """Phase 5: Start daily rebalance scheduler daemon."""
    from src.alerts.scheduler import start_scheduler
    start_scheduler(
        portfolio_value=value,
        config_path=config,
        hour=hour,
        minute=minute,
    )


@app.command()
def run_sweep(
    config: str = typer.Option("config.yaml", help="Path to config.yaml"),
):
    """Run parameter sweep across weight_max and rebalance_frequency."""
    import subprocess
    subprocess.run(["python", "run_parameter_sweep.py"])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
