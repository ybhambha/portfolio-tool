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
# Fidelity personal portfolio
# ---------------------------------------------------------------------------

@app.command()
def fidelity_connect(
    config: str = typer.Option("config.yaml", help="Path to config.yaml"),
    reconnect: str = typer.Option(None, help="Connection id to re-authorize (if it expired)"),
):
    """Print a SnapTrade link to connect your Fidelity login (read-only)."""
    from src.config import load_config
    from src.fidelity.snaptrade_source import SnapTradeFidelity
    cfg = load_config(config)
    st = SnapTradeFidelity(broker_slug=cfg.fidelity.snaptrade_broker)
    print("\nOpen this link, choose Fidelity and log in (read-only access):\n")
    print(st.connection_link(reconnect=reconnect))
    print("\nThen run: python -m src.main fidelity-report\n")


@app.command()
def fidelity_positions(
    config: str = typer.Option("config.yaml", help="Path to config.yaml"),
    source: str = typer.Option(None, help="snaptrade | csv (default: config)"),
    positions_csv: str = typer.Option(None, help="Explicit Fidelity positions CSV"),
):
    """Show current Fidelity positions across all accounts."""
    import pandas as pd
    from src.config import load_config
    from src.fidelity.pipeline import load_fidelity_data
    cfg = load_config(config)
    pos, acts, label = load_fidelity_data(cfg, source, positions_csv)
    total = pos["market_value"].sum()
    view = pos.assign(weight=pos["market_value"] / total)[
        ["account_name", "account_type", "ticker", "quantity", "price", "market_value", "cost_basis", "weight"]]
    with pd.option_context("display.max_rows", 500, "display.width", 160,
                           "display.float_format", "{:,.2f}".format):
        print(f"\nSource: {label}\n")
        print(view.sort_values("market_value", ascending=False).to_string(index=False))
    print(f"\nTotal: ${total:,.2f} across {pos['account_id'].nunique()} account(s); "
          f"{len(acts)} transactions loaded")


@app.command()
def fidelity_report(
    config: str = typer.Option("config.yaml", help="Path to config.yaml"),
    source: str = typer.Option(None, help="snaptrade | csv (default: config)"),
    positions_csv: str = typer.Option(None, help="Explicit Fidelity positions CSV"),
    history_csv: list[str] = typer.Option(None, help="Fidelity history CSV(s); repeat the flag for several"),
    rebalance: bool = typer.Option(True, help="Include the rebalance trade list"),
    output: str = typer.Option(None, help="Output HTML path"),
):
    """Performance + risk report and rebalance trade list for your Fidelity accounts."""
    import os
    from src.config import load_config
    from src.fidelity.pipeline import run_fidelity
    cfg = load_config(config)
    res = run_fidelity(cfg, source=source, positions_csv=positions_csv,
                       history_csv=history_csv or None, rebalance=rebalance, output=output)
    r, p = res["realized"], res["plan"]
    print(f"\nSource: {res['source']} | value ${res['positions']['market_value'].sum():,.0f}")
    if r:
        mwr = f"{r['mwr']:.2%}" if r["mwr"] is not None else "n/a"
        print(f"TWR {r['twr']:.2%} (bench {r['bench_twr']:.2%}) | money-weighted {mwr} | "
              f"since {r['start']:%Y-%m-%d}")
    if p is not None:
        s = p["summary"]
        print(f"Rebalance: {s['n_trades']} trades | turnover {s['turnover']:.1%} | "
              f"est. taxable gain ${s['est_realized_gain']:,.0f}")
        if not p["trades"].empty:
            print(p["trades"][["account_name", "action", "ticker", "shares", "est_value", "note"]]
                  .to_string(index=False))
    for w in res["warnings"]:
        print(f"  note: {w}")
    print(f"\nReport: {os.path.abspath(res['report'])}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
