"""
report.py
---------
Generates a clean, self-contained HTML performance report
combining all metrics, charts, and year-by-year breakdown.

The report is saved to data/reports/performance_report.html
and can be opened in any web browser.
"""

import os
import base64
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for report generation
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import TwoSlopeNorm

from src.backtest.metrics import (
    compute_cagr,
    compute_volatility,
    compute_sharpe,
    compute_drawdown_series,
    compute_max_drawdown,
    compute_avg_drawdown,
    compute_calmar,
    compute_monthly_returns,
    compute_all_metrics,
    compute_win_rate,
)

import logging
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chart generation (returns base64 encoded images for embedding in HTML)
# ---------------------------------------------------------------------------

def _fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 string for HTML embedding."""
    from io import BytesIO
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="white")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_base64


def _chart_nav(portfolio_nav, benchmark_nav) -> str:
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(portfolio_nav.index, portfolio_nav.values,
            label="MVO Portfolio", color="#2196F3", linewidth=1.5)
    ax.plot(benchmark_nav.index, benchmark_nav.values,
            label="SPY (Buy & Hold)", color="#FF9800",
            linewidth=1.5, linestyle="--")
    ax.set_title("Portfolio NAV vs SPY", fontsize=13, fontweight="bold")
    ax.set_ylabel("NAV ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    return _fig_to_base64(fig)


def _chart_drawdown(portfolio_nav, benchmark_nav) -> str:
    port_dd = compute_drawdown_series(portfolio_nav) * 100
    bench_dd = compute_drawdown_series(benchmark_nav) * 100
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.fill_between(port_dd.index, port_dd.values, 0,
                    alpha=0.4, color="#2196F3", label="MVO Portfolio")
    ax.fill_between(bench_dd.index, bench_dd.values, 0,
                    alpha=0.3, color="#FF9800", label="SPY")
    ax.plot(port_dd.index, port_dd.values, color="#2196F3", linewidth=0.8)
    ax.plot(bench_dd.index, bench_dd.values, color="#FF9800",
            linewidth=0.8, linestyle="--")
    ax.set_title("Drawdown", fontsize=13, fontweight="bold")
    ax.set_ylabel("Drawdown (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    return _fig_to_base64(fig)


def _chart_rolling_sharpe(portfolio_returns, benchmark_returns,
                           risk_free_rate=0.05) -> str:
    window = 252
    daily_rf = risk_free_rate / 252
    port_excess = portfolio_returns - daily_rf
    bench_excess = benchmark_returns - daily_rf
    port_sharpe = (port_excess.rolling(window).mean() /
                   port_excess.rolling(window).std()) * np.sqrt(252)
    bench_sharpe = (bench_excess.rolling(window).mean() /
                    bench_excess.rolling(window).std()) * np.sqrt(252)
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(port_sharpe.index, port_sharpe.values,
            label="MVO Portfolio", color="#2196F3", linewidth=1.5)
    ax.plot(bench_sharpe.index, bench_sharpe.values,
            label="SPY", color="#FF9800", linewidth=1.5, linestyle="--")
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.axhline(y=1, color="green", linewidth=0.8, linestyle=":",
               alpha=0.5, label="Sharpe = 1")
    ax.set_title("Rolling 12-Month Sharpe Ratio", fontsize=13, fontweight="bold")
    ax.set_ylabel("Sharpe Ratio")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    return _fig_to_base64(fig)


def _chart_heatmap(portfolio_returns) -> str:
    monthly = compute_monthly_returns(portfolio_returns) * 100
    monthly_df = monthly.to_frame("return")
    monthly_df["year"] = monthly_df.index.year
    monthly_df["month"] = monthly_df.index.month
    pivot = monthly_df.pivot(index="year", columns="month", values="return")
    pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig, ax = plt.subplots(figsize=(13, len(pivot) * 0.7 + 1))
    vmin = pivot.min().min()
    vmax = pivot.max().max()
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    im = ax.imshow(pivot.values, cmap="RdYlGn", norm=norm, aspect="auto")
    ax.set_xticks(range(12))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                        fontsize=8,
                        color="black" if abs(val) < 8 else "white")
    plt.colorbar(im, ax=ax, label="Monthly Return (%)")
    ax.set_title("Portfolio Monthly Returns Heatmap",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    return _fig_to_base64(fig)


# ---------------------------------------------------------------------------
# Year-by-year breakdown
# ---------------------------------------------------------------------------

def compute_yearly_breakdown(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.05,
) -> pd.DataFrame:
    """Compute annual return, volatility and Sharpe for each year."""
    rows = []
    years = sorted(portfolio_returns.index.year.unique())

    for year in years:
        p = portfolio_returns[portfolio_returns.index.year == year]
        b = benchmark_returns[benchmark_returns.index.year == year]

        if len(p) < 20:
            continue

        p_ret  = round((1 + p).prod() - 1, 4) * 100
        b_ret  = round((1 + b).prod() - 1, 4) * 100
        p_vol  = round(p.std() * np.sqrt(252) * 100, 2)
        p_sh   = round((p.mean() * 252 - risk_free_rate) /
                       (p.std() * np.sqrt(252)), 3)

        rows.append({
            "Year"              : year,
            "Portfolio Return%" : round(p_ret, 2),
            "SPY Return%"       : round(b_ret, 2),
            "Alpha%"            : round(p_ret - b_ret, 2),
            "Portfolio Vol%"    : p_vol,
            "Portfolio Sharpe"  : p_sh,
        })

    return pd.DataFrame(rows).set_index("Year")


# ---------------------------------------------------------------------------
# HTML report generation
# ---------------------------------------------------------------------------

def _metric_card(label, portfolio_val, spy_val, diff, better_higher=True):
    """Generate HTML for a single metric card."""
    try:
        diff_num = float(str(diff).replace("%", "").replace("+", ""))
        diff_color = "#4CAF50" if (diff_num > 0) == better_higher else "#f44336"
    except (ValueError, TypeError):
        diff_color = "#333"
    diff_sign = "+" if str(diff).replace("%","").replace("+","") and float(str(diff).replace("%","").replace("+","")) > 0 else ""
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-row">
            <div class="metric-col">
                <div class="metric-sub">Portfolio</div>
                <div class="metric-val">{portfolio_val}</div>
            </div>
            <div class="metric-col">
                <div class="metric-sub">SPY</div>
                <div class="metric-val spy">{spy_val}</div>
            </div>
            <div class="metric-col">
                <div class="metric-sub">Diff</div>
                <div class="metric-val" style="color:{diff_color}">
                    {diff_sign}{diff}
                </div>
            </div>
        </div>
    </div>"""


def generate_html_report(
    portfolio_nav: pd.Series,
    benchmark_nav: pd.Series,
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    weights_df: pd.DataFrame,
    config,
    output_path: str = "data/reports/performance_report.html",
) -> str:
    """
    Generate a self-contained HTML performance report.

    Parameters
    ----------
    portfolio_nav       : daily portfolio NAV series
    benchmark_nav       : daily benchmark NAV series
    portfolio_returns   : daily portfolio returns
    benchmark_returns   : daily benchmark returns
    weights_df          : rebalance weights DataFrame
    config              : AppConfig object
    output_path         : where to save the HTML file

    Returns
    -------
    str : path to saved report
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rfr = config.performance.risk_free_rate

    # --- Compute metrics ---
    metrics_df   = compute_all_metrics(
        portfolio_nav, portfolio_returns,
        benchmark_nav, benchmark_returns, rfr
    )
    win_rate     = compute_win_rate(portfolio_returns, benchmark_returns)
    yearly_df    = compute_yearly_breakdown(portfolio_returns,
                                            benchmark_returns, rfr)
    avg_turnover = weights_df.diff().abs().sum(axis=1).dropna().mean() * 100

    # --- Generate charts ---
    logger.info("Generating report charts...")
    print("  Generating NAV chart...")
    nav_img     = _chart_nav(portfolio_nav, benchmark_nav)
    print("  Generating drawdown chart...")
    dd_img      = _chart_drawdown(portfolio_nav, benchmark_nav)
    print("  Generating rolling Sharpe chart...")
    sharpe_img  = _chart_rolling_sharpe(portfolio_returns,
                                        benchmark_returns, rfr)
    print("  Generating monthly heatmap...")
    heatmap_img = _chart_heatmap(portfolio_returns)

    # --- Helper values ---
    start_date  = portfolio_nav.index[0].strftime("%Y-%m-%d")
    end_date    = portfolio_nav.index[-1].strftime("%Y-%m-%d")
    gen_date    = datetime.now().strftime("%B %d, %Y %H:%M")
    port_final  = f"${portfolio_nav.iloc[-1]:,.2f}"
    spy_final   = f"${benchmark_nav.iloc[-1]:,.2f}"
    diff_final  = f"${portfolio_nav.iloc[-1] - benchmark_nav.iloc[-1]:,.2f}"

    m = metrics_df  # shorthand

    # --- Yearly table rows ---
    yearly_rows = ""
    for year, row in yearly_df.iterrows():
        alpha_color = "#4CAF50" if row["Alpha%"] > 0 else "#f44336"
        yearly_rows += f"""
        <tr>
            <td>{year}</td>
            <td>{row['Portfolio Return%']}%</td>
            <td>{row['SPY Return%']}%</td>
            <td style="color:{alpha_color};font-weight:500">
                {'+' if row['Alpha%'] > 0 else ''}{row['Alpha%']}%
            </td>
            <td>{row['Portfolio Vol%']}%</td>
            <td>{row['Portfolio Sharpe']}</td>
        </tr>"""

    # --- Avg weights table ---
    avg_weights = weights_df.mean().sort_values(ascending=False)
    weight_rows = ""
    for ticker, w in avg_weights.items():
        bar_width = int(w * 300)
        weight_rows += f"""
        <tr>
            <td>{ticker}</td>
            <td>{w*100:.2f}%</td>
            <td>
                <div style="background:#2196F3;height:12px;
                            width:{bar_width}px;border-radius:2px;">
                </div>
            </td>
        </tr>"""

    # --- HTML template ---
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Portfolio Performance Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI',
          sans-serif; background: #f5f6fa; color: #333; }}
  .header {{ background: #1a237e; color: white; padding: 2rem 2.5rem; }}
  .header h1 {{ font-size: 1.8rem; font-weight: 600; }}
  .header p {{ opacity: 0.8; margin-top: 0.4rem; font-size: 0.9rem; }}
  .container {{ max-width: 1100px; margin: 0 auto; padding: 2rem 1.5rem; }}
  .section {{ background: white; border-radius: 10px; padding: 1.5rem;
              margin-bottom: 1.5rem;
              box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
  .section h2 {{ font-size: 1.1rem; font-weight: 600; color: #1a237e;
                 margin-bottom: 1.2rem; padding-bottom: 0.5rem;
                 border-bottom: 2px solid #e8eaf6; }}
  .summary-grid {{ display: grid;
                   grid-template-columns: repeat(3, 1fr);
                   gap: 1rem; margin-bottom: 1.5rem; }}
  .summary-card {{ background: #e8eaf6; border-radius: 8px;
                   padding: 1rem 1.25rem; text-align: center; }}
  .summary-card .label {{ font-size: 0.75rem; color: #555;
                           text-transform: uppercase; letter-spacing: 0.05em; }}
  .summary-card .value {{ font-size: 1.5rem; font-weight: 600;
                           color: #1a237e; margin-top: 0.3rem; }}
  .summary-card .sub {{ font-size: 0.8rem; color: #777; margin-top: 0.2rem; }}
  .metrics-grid {{ display: grid; grid-template-columns: repeat(3, 1fr);
                   gap: 1rem; }}
  .metric-card {{ background: #fafafa; border: 1px solid #e0e0e0;
                  border-radius: 8px; padding: 1rem; }}
  .metric-label {{ font-size: 0.78rem; color: #666; font-weight: 500;
                   text-transform: uppercase; letter-spacing: 0.04em;
                   margin-bottom: 0.6rem; }}
  .metric-row {{ display: flex; gap: 0.5rem; }}
  .metric-col {{ flex: 1; text-align: center; }}
  .metric-sub {{ font-size: 0.7rem; color: #999; margin-bottom: 0.2rem; }}
  .metric-val {{ font-size: 1rem; font-weight: 600; color: #333; }}
  .metric-val.spy {{ color: #FF9800; }}
  img {{ width: 100%; border-radius: 6px; margin-top: 0.5rem; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
  th {{ background: #e8eaf6; color: #1a237e; padding: 0.6rem 0.8rem;
        text-align: left; font-weight: 600; }}
  td {{ padding: 0.5rem 0.8rem; border-bottom: 1px solid #f0f0f0; }}
  tr:hover td {{ background: #fafafa; }}
  .config-grid {{ display: grid; grid-template-columns: repeat(2, 1fr);
                  gap: 0.5rem; font-size: 0.9rem; }}
  .config-row {{ display: flex; justify-content: space-between;
                 padding: 0.4rem 0.6rem; background: #fafafa;
                 border-radius: 4px; }}
  .config-key {{ color: #666; }}
  .config-val {{ font-weight: 500; }}
  .footer {{ text-align: center; color: #999; font-size: 0.8rem;
             padding: 1.5rem; }}
</style>
</head>
<body>

<div class="header">
  <h1>Portfolio Performance Report</h1>
  <p>MVO Sector ETF Strategy vs SPY &nbsp;|&nbsp;
     {start_date} → {end_date} &nbsp;|&nbsp;
     Generated {gen_date}</p>
</div>

<div class="container">

  <!-- Summary Cards -->
  <div class="section">
    <h2>Summary</h2>
    <div class="summary-grid">
      <div class="summary-card">
        <div class="label">Portfolio Final NAV</div>
        <div class="value">{port_final}</div>
        <div class="sub">Starting from $10,000</div>
      </div>
      <div class="summary-card">
        <div class="label">SPY Final NAV</div>
        <div class="value">{spy_final}</div>
        <div class="sub">Buy & Hold Benchmark</div>
      </div>
      <div class="summary-card">
        <div class="label">Outperformance</div>
        <div class="value" style="color:#4CAF50">{diff_final}</div>
        <div class="sub">Portfolio minus SPY</div>
      </div>
    </div>
  </div>

  <!-- Performance Metrics -->
  <div class="section">
    <h2>Performance Metrics</h2>
    <div class="metrics-grid">
      {_metric_card(
            "Annualized Return",
            f"{m.loc['Annualized Return (%)', 'Portfolio']}%",
            f"{m.loc['Annualized Return (%)', 'SPY']}%",
            f"{round(m.loc['Annualized Return (%)', 'Difference'], 2)}%",
            better_higher=True
        )}
        {_metric_card(
            "Annualized Volatility",
            f"{m.loc['Annualized Volatility (%)', 'Portfolio']}%",
            f"{m.loc['Annualized Volatility (%)', 'SPY']}%",
            f"{round(m.loc['Annualized Volatility (%)', 'Difference'], 2)}%",
            better_higher=False
        )}
        {_metric_card(
            "Sharpe Ratio",
            f"{m.loc['Sharpe Ratio', 'Portfolio']}",
            f"{m.loc['Sharpe Ratio', 'SPY']}",
            f"{round(m.loc['Sharpe Ratio', 'Difference'], 3)}",
            better_higher=True
        )}
        {_metric_card(
            "Max Drawdown",
            f"{m.loc['Max Drawdown (%)', 'Portfolio']}%",
            f"{m.loc['Max Drawdown (%)', 'SPY']}%",
            f"{round(m.loc['Max Drawdown (%)', 'Difference'], 2)}%",
            better_higher=True
        )}
        {_metric_card(
            "Avg Drawdown",
            f"{m.loc['Avg Drawdown (%)', 'Portfolio']}%",
            f"{m.loc['Avg Drawdown (%)', 'SPY']}%",
            f"{round(m.loc['Avg Drawdown (%)', 'Difference'], 2)}%",
            better_higher=True
        )}
        {_metric_card(
            "Calmar Ratio",
            f"{m.loc['Calmar Ratio', 'Portfolio']}",
            f"{m.loc['Calmar Ratio', 'SPY']}",
            f"{round(m.loc['Calmar Ratio', 'Difference'], 3)}",
            better_higher=True
        )}
    </div>
    <div style="margin-top:1rem; padding:0.8rem 1rem; background:#e8f5e9;
                border-radius:6px; font-size:0.9rem;">
      Monthly Win Rate vs SPY: <strong>{win_rate:.1f}%</strong>
      &nbsp;|&nbsp; Avg Monthly Turnover:
      <strong>{avg_turnover:.1f}%</strong>
    </div>
  </div>

  <!-- NAV Chart -->
  <div class="section">
    <h2>NAV Growth</h2>
    <img src="data:image/png;base64,{nav_img}" alt="NAV Chart">
  </div>

  <!-- Drawdown Chart -->
  <div class="section">
    <h2>Drawdown</h2>
    <img src="data:image/png;base64,{dd_img}" alt="Drawdown Chart">
  </div>

  <!-- Rolling Sharpe Chart -->
  <div class="section">
    <h2>Rolling 12-Month Sharpe Ratio</h2>
    <img src="data:image/png;base64,{sharpe_img}" alt="Rolling Sharpe Chart">
  </div>

  <!-- Monthly Heatmap -->
  <div class="section">
    <h2>Monthly Returns Heatmap</h2>
    <img src="data:image/png;base64,{heatmap_img}" alt="Monthly Heatmap">
  </div>

  <!-- Year by Year -->
  <div class="section">
    <h2>Year-by-Year Breakdown</h2>
    <table>
      <thead>
        <tr>
          <th>Year</th>
          <th>Portfolio Return</th>
          <th>SPY Return</th>
          <th>Alpha</th>
          <th>Portfolio Vol</th>
          <th>Portfolio Sharpe</th>
        </tr>
      </thead>
      <tbody>{yearly_rows}</tbody>
    </table>
  </div>

  <!-- Average Weights -->
  <div class="section">
    <h2>Average Portfolio Weights (over backtest period)</h2>
    <table>
      <thead>
        <tr><th>ETF</th><th>Avg Weight</th><th>Allocation</th></tr>
      </thead>
      <tbody>{weight_rows}</tbody>
    </table>
  </div>

  <!-- Strategy Config -->
  <div class="section">
    <h2>Strategy Configuration</h2>
    <div class="config-grid">
      <div class="config-row">
        <span class="config-key">Universe</span>
        <span class="config-val">{', '.join(config.universe.etfs)}</span>
      </div>
      <div class="config-row">
        <span class="config-key">Benchmark</span>
        <span class="config-val">{config.universe.benchmark}</span>
      </div>
      <div class="config-row">
        <span class="config-key">Lookback window</span>
        <span class="config-val">{config.optimizer.lookback_days} days</span>
      </div>
      <div class="config-row">
        <span class="config-key">Rebalance frequency</span>
        <span class="config-val">{config.optimizer.rebalance_frequency}</span>
      </div>
      <div class="config-row">
        <span class="config-key">Max weight per ETF</span>
        <span class="config-val">
            {int(config.optimizer.weight_max * 100)}%
        </span>
      </div>
      <div class="config-row">
        <span class="config-key">Min weight per ETF</span>
        <span class="config-val">
            {int(config.optimizer.weight_min * 100)}%
        </span>
      </div>
      <div class="config-row">
        <span class="config-key">Transaction cost</span>
        <span class="config-val">
            {config.optimizer.transaction_cost_bps} bps
        </span>
      </div>
      <div class="config-row">
        <span class="config-key">Risk-free rate</span>
        <span class="config-val">
            {config.performance.risk_free_rate * 100:.1f}%
        </span>
      </div>
      <div class="config-row">
        <span class="config-key">Covariance method</span>
        <span class="config-val">Ledoit-Wolf shrinkage</span>
      </div>
      <div class="config-row">
        <span class="config-key">Optimization method</span>
        <span class="config-val">Mean-Variance (Markowitz)</span>
      </div>
    </div>
  </div>

</div>

<div class="footer">
  Generated by Portfolio Construction Tool &nbsp;|&nbsp; {gen_date}
</div>

</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info(f"Report saved to {output_path}")
    return output_path
