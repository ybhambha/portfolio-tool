"""
signals.py
----------
Generates rebalance signals and trade lists.

On each rebalance date:
  1. Fetches latest prices from yfinance
  2. Computes new optimal weights using the past 252 days
  3. Compares new weights vs current weights
  4. Generates a trade list with dollar amounts to buy/sell
  5. Estimates transaction costs

The trade list is passed to the email module for delivery.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import logging
from datetime import datetime, date

from src.optimizer.covariance import compute_covariance, compute_expected_returns
from src.optimizer.optimizer import optimize_portfolio, compute_portfolio_metrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rebalance date check
# ---------------------------------------------------------------------------

def is_rebalance_day(
    frequency: str = "monthly",
    reference_date: date | None = None,
) -> bool:
    """
    Check if today (or reference_date) is a rebalance day.

    For monthly: last trading day of the month
    For quarterly: last trading day of March, June, September, December

    Parameters
    ----------
    frequency      : "monthly" or "quarterly"
    reference_date : date to check (defaults to today)

    Returns
    -------
    bool : True if today is a rebalance day
    """
    check_date = reference_date or date.today()
    check_ts   = pd.Timestamp(check_date)

    if frequency == "monthly":
        # Last business day of the month
        last_bday = pd.offsets.BMonthEnd().rollforward(check_ts)
        return check_ts == last_bday

    elif frequency == "quarterly":
        quarter_end_months = [3, 6, 9, 12]
        if check_ts.month not in quarter_end_months:
            return False
        last_bday = pd.offsets.BMonthEnd().rollforward(check_ts)
        return check_ts == last_bday

    else:
        raise ValueError(f"Unknown frequency: {frequency}")


def days_until_next_rebalance(frequency: str = "monthly") -> int:
    """Return number of calendar days until next rebalance date."""
    today = pd.Timestamp(date.today())

    if frequency == "monthly":
        next_rebal = pd.offsets.BMonthEnd().rollforward(today)
        if next_rebal == today:
            next_rebal = (today + pd.offsets.BMonthEnd(1))
    elif frequency == "quarterly":
        next_rebal = pd.offsets.BQuarterEnd().rollforward(today)
        if next_rebal == today:
            next_rebal = (today + pd.offsets.BQuarterEnd(1))
    else:
        raise ValueError(f"Unknown frequency: {frequency}")

    return (next_rebal - today).days


# ---------------------------------------------------------------------------
# Fetch latest data
# ---------------------------------------------------------------------------

def fetch_latest_returns(
    tickers: list[str],
    lookback_days: int = 252,
    price_col: str = "Adj Close",
) -> pd.DataFrame:
    """
    Fetch the most recent `lookback_days` of log returns from yfinance.

    Parameters
    ----------
    tickers      : list of ETF tickers
    lookback_days: number of trading days to fetch
    price_col    : which price column to use

    Returns
    -------
    pd.DataFrame : log returns, shape (lookback_days, n_tickers)
    """
    # Fetch extra days to account for weekends/holidays
    calendar_days = int(lookback_days * 1.5)
    end   = pd.Timestamp.today()
    start = end - pd.Timedelta(days=calendar_days)

    logger.info(f"Fetching latest prices for {tickers}...")
    raw = yf.download(
        tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
    )

    # Handle MultiIndex columns from yfinance
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw[price_col]
    else:
        prices = raw[[price_col]].rename(columns={price_col: tickers[0]})

    prices = prices.dropna()
    log_returns = np.log(prices / prices.shift(1)).dropna()

    # Keep only the most recent lookback_days
    log_returns = log_returns.tail(lookback_days)

    logger.info(f"Fetched {len(log_returns)} days of returns")
    return log_returns


def fetch_current_prices(tickers: list[str]) -> pd.Series:
    """Fetch the most recent closing price for each ticker."""
    raw = yf.download(
        tickers,
        period="5d",
        auto_adjust=False,
        progress=False,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Adj Close"].iloc[-1]
    else:
        prices = raw["Adj Close"].iloc[-1:].squeeze()

    return prices


# ---------------------------------------------------------------------------
# Trade list generation
# ---------------------------------------------------------------------------

def generate_trade_list(
    current_weights: dict[str, float],
    target_weights: pd.Series,
    portfolio_value: float,
    transaction_cost_bps: float = 5.0,
) -> pd.DataFrame:
    """
    Generate a trade list comparing current vs target weights.

    Parameters
    ----------
    current_weights     : dict of {ticker: current_weight}
    target_weights      : pd.Series of optimal weights from optimizer
    portfolio_value     : total portfolio value in dollars
    transaction_cost_bps: cost per trade in basis points

    Returns
    -------
    pd.DataFrame : trade list with columns:
        ticker, current_weight, target_weight, weight_change,
        current_value, target_value, trade_amount, action, est_cost
    """
    tickers = target_weights.index.tolist()
    rows    = []

    for ticker in tickers:
        curr_w  = current_weights.get(ticker, 0.0)
        tgt_w   = target_weights[ticker]
        delta_w = tgt_w - curr_w

        curr_val  = curr_w  * portfolio_value
        tgt_val   = tgt_w   * portfolio_value
        trade_amt = delta_w * portfolio_value
        est_cost  = abs(trade_amt) * (transaction_cost_bps / 10_000)

        if abs(delta_w) < 0.001:
            action = "HOLD"
        elif trade_amt > 0:
            action = "BUY"
        else:
            action = "SELL"

        rows.append({
            "Ticker"          : ticker,
            "Current Weight"  : round(curr_w  * 100, 2),
            "Target Weight"   : round(tgt_w   * 100, 2),
            "Weight Change"   : round(delta_w * 100, 2),
            "Current Value $" : round(curr_val,  2),
            "Target Value $"  : round(tgt_val,   2),
            "Trade Amount $"  : round(trade_amt,  2),
            "Action"          : action,
            "Est. Cost $"     : round(est_cost,   2),
        })

    df = pd.DataFrame(rows).sort_values("Trade Amount $")
    return df


# ---------------------------------------------------------------------------
# Main signal generator
# ---------------------------------------------------------------------------

def generate_rebalance_signal(
    config,
    portfolio_value: float = 10_000.0,
    current_weights: dict | None = None,
) -> dict:
    """
    Generate a full rebalance signal including new weights and trade list.

    Parameters
    ----------
    config           : AppConfig object
    portfolio_value  : current total portfolio value in dollars
    current_weights  : dict of current weights {ticker: weight}
                       If None, assumes equal weights

    Returns
    -------
    dict with keys:
        signal_date    : today's date
        is_rebalance   : bool
        target_weights : pd.Series
        trade_list     : pd.DataFrame
        metrics        : dict of expected portfolio metrics
        days_to_next   : int
        summary        : str (human readable summary)
    """
    today         = date.today()
    rebal_freq    = config.optimizer.rebalance_frequency
    is_rebal      = is_rebalance_day(rebal_freq)
    days_to_next  = days_until_next_rebalance(rebal_freq)
    etfs          = config.universe.etfs
    rfr           = config.performance.risk_free_rate

    if not is_rebal:
        return {
            "signal_date" : today,
            "is_rebalance": False,
            "days_to_next": days_to_next,
            "summary"     : (
                f"No rebalance today ({today}). "
                f"Next rebalance in {days_to_next} days."
            ),
        }

    # Default to equal weights if no current weights provided
    if current_weights is None:
        current_weights = {t: 1/len(etfs) for t in etfs}

    # Fetch latest returns
    log_returns = fetch_latest_returns(
        tickers=etfs,
        lookback_days=config.optimizer.lookback_days,
    )

    # Compute optimal weights
    mu  = compute_expected_returns(log_returns, annualize=True)
    cov = compute_covariance(log_returns, method="ledoit_wolf", annualize=True)

    current_w_series = pd.Series(current_weights).reindex(etfs).fillna(0)

    target_weights = optimize_portfolio(
        mu=mu,
        cov=cov,
        current_weights=current_w_series,
        weight_min=config.optimizer.weight_min,
        weight_max=config.optimizer.weight_max,
        transaction_cost_bps=config.optimizer.transaction_cost_bps,
    )

    # Portfolio metrics
    metrics = compute_portfolio_metrics(
        target_weights, mu, cov, risk_free_rate=rfr
    )

    # Trade list
    trade_list = generate_trade_list(
        current_weights=current_weights,
        target_weights=target_weights,
        portfolio_value=portfolio_value,
        transaction_cost_bps=config.optimizer.transaction_cost_bps,
    )

    total_cost    = trade_list["Est. Cost $"].sum()
    n_trades      = (trade_list["Action"] != "HOLD").sum()
    next_rebal    = days_until_next_rebalance(rebal_freq)

    summary = (
        f"REBALANCE SIGNAL — {today.strftime('%B %d, %Y')}\n"
        f"Portfolio Value : ${portfolio_value:,.2f}\n"
        f"Trades required : {n_trades}\n"
        f"Estimated cost  : ${total_cost:.2f}\n"
        f"Expected return : {metrics['expected_return']}%\n"
        f"Expected Sharpe : {metrics['sharpe_ratio']}\n"
        f"Next rebalance  : in {next_rebal} days"
    )

    logger.info(summary)

    return {
        "signal_date"   : today,
        "is_rebalance"  : True,
        "target_weights": target_weights,
        "trade_list"    : trade_list,
        "metrics"       : metrics,
        "days_to_next"  : next_rebal,
        "summary"       : summary,
        "total_cost"    : total_cost,
        "n_trades"      : n_trades,
    }
