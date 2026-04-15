# src/backtest/__init__.py
from src.backtest.simulator import simulate_portfolio, simulate_benchmark
from src.backtest.metrics import (
    compute_all_metrics,
    compute_win_rate,
    print_metrics_table,
    compute_drawdown_series,
    compute_monthly_returns,
)
from src.backtest.charts import plot_all
