"""
performance.py
--------------
Two views of performance for the Fidelity portfolio:

1. Holdings analytics — "how have today's holdings behaved?"
   Current weights applied to trailing price history: CAGR, vol, Sharpe,
   drawdown, beta vs benchmark, and each holding's contribution to risk.

2. Realized performance — "how did I actually do?"
   Daily holdings are reconstructed *backwards* from current positions using
   the transaction history, so an incomplete history still reconciles to
   today's actual share counts. From that:
     - Time-weighted return (TWR)   — manager skill, removes flow timing
     - Money-weighted return (XIRR) — investor experience, includes timing
     - Public-market equivalent vs benchmark: the same cash flows invested
       in SPY instead.
   Scope is the invested securities sleeve (cash excluded); dividends and
   interest on holdings count as return.
"""

from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from src.backtest.metrics import (
    compute_all_metrics, compute_max_drawdown, compute_drawdown_series,
)

logger = logging.getLogger(__name__)

_SHARE_SIGN = {"BUY": 1, "REI": 1, "TRANSFER_IN": 1, "STOCK_DIV": 1,
               "SELL": -1, "TRANSFER_OUT": -1}
_INCOME = {"DIVIDEND", "INTEREST"}


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------

def fetch_market_data(tickers: list[str], start, end=None) -> dict[str, pd.DataFrame]:
    """
    Download from yfinance. Returns
      adj_close : total-return prices (for analytics / benchmark)
      close     : split-adjusted, NOT dividend-adjusted (for valuing share counts)
      splits    : split ratios (0 where none)
    """
    import yfinance as yf

    # Brokerage "BRK.B" → Yahoo "BRK-B"; map back afterwards
    to_yf = {t: (t.replace(".", "-") if t.count(".") == 1 and len(t) <= 6 else t) for t in set(tickers)}
    from_yf = {v: k for k, v in to_yf.items()}
    tickers = sorted(to_yf.values())
    raw = yf.download(tickers, start=pd.Timestamp(start).strftime("%Y-%m-%d"),
                      end=None if end is None else pd.Timestamp(end).strftime("%Y-%m-%d"),
                      auto_adjust=False, actions=True, progress=False, group_by="column")
    if raw.empty:
        raise RuntimeError("yfinance returned no data")
    if not isinstance(raw.columns, pd.MultiIndex):
        raw.columns = pd.MultiIndex.from_product([raw.columns, tickers])

    def field(name):
        return raw[name] if name in raw.columns.get_level_values(0) else pd.DataFrame(index=raw.index)

    out = {
        "adj_close": field("Adj Close").dropna(how="all", axis=1).ffill(),
        "close": field("Close").dropna(how="all", axis=1).ffill(),
        "splits": field("Stock Splits").fillna(0.0),
    }
    for k in out:
        out[k] = out[k].rename(columns=from_yf)
        out[k].index = pd.to_datetime(out[k].index).tz_localize(None)
    missing = sorted(set(to_yf) - set(out["close"].columns))
    if missing:
        logger.warning(f"No price history for: {missing} — excluded from analytics")
    return out


# ---------------------------------------------------------------------------
# 1. Holdings analytics
# ---------------------------------------------------------------------------

def current_weights(positions: pd.DataFrame, include_cash: bool = True) -> pd.Series:
    df = positions if include_cash else positions[positions["asset_kind"] != "cash"]
    mv = df.assign(ticker=np.where(df["asset_kind"] == "cash", "CASH", df["ticker"])) \
           .groupby("ticker")["market_value"].sum()
    return (mv / mv.sum()).sort_values(ascending=False)


def holdings_analytics(
    positions: pd.DataFrame,
    adj_close: pd.DataFrame,
    benchmark: str = "SPY",
    risk_free_rate: float = 0.05,
    lookback_days: int = 756,
) -> dict:
    w = current_weights(positions)
    priced = [t for t in w.index if t in adj_close.columns and t != "CASH"]
    unpriced = [t for t in w.index if t not in priced and t != "CASH"]
    w_used = w[priced + (["CASH"] if "CASH" in w.index else [])]
    w_used = w_used / w_used.sum()

    px = adj_close[priced + [benchmark]].tail(lookback_days + 1)
    rets = px.pct_change().iloc[1:]
    rets[priced] = rets[priced].fillna(0.0)
    if "CASH" in w_used.index:
        rets["CASH"] = risk_free_rate / 252
    port = rets[w_used.index] @ w_used.values
    bench = rets[benchmark].fillna(0.0)

    port_nav, bench_nav = (1 + port).cumprod(), (1 + bench).cumprod()
    table = compute_all_metrics(port_nav, port, bench_nav, bench, risk_free_rate)
    table = table.rename(columns={"SPY": benchmark})

    beta = float(np.cov(port, bench)[0, 1] / bench.var())
    tracking_error = float((port - bench).std() * np.sqrt(252))

    # Risk contribution (annualized covariance of the priced sleeve)
    sec = [t for t in w_used.index if t != "CASH"]
    cov = rets[sec].cov() * 252
    ws = w_used[sec]
    port_var = float(ws @ cov @ ws)
    mctr = cov @ ws
    risk = pd.DataFrame({
        "weight": ws,
        "ann_vol": np.sqrt(np.diag(cov)),
        "beta": [float(np.cov(rets[t], bench)[0, 1] / bench.var()) for t in sec],
        "risk_contrib": ws * mctr / port_var if port_var > 0 else 0.0,
    }).sort_values("risk_contrib", ascending=False)

    return {
        "weights": w,
        "metrics": table,
        "beta": beta,
        "tracking_error": tracking_error,
        "risk": risk,
        "nav": pd.DataFrame({"Portfolio": port_nav, benchmark: bench_nav}),
        "unpriced": unpriced,
        "window": (rets.index[0], rets.index[-1]),
    }


# ---------------------------------------------------------------------------
# 2. Realized performance from transactions
# ---------------------------------------------------------------------------

def _split_factor_after(splits: pd.Series, when: pd.Timestamp) -> float:
    s = splits[(splits.index > when) & (splits > 0)]
    return float(s.prod()) if len(s) else 1.0


def reconstruct_history(
    positions: pd.DataFrame,
    activities: pd.DataFrame,
    close: pd.DataFrame,
    splits: pd.DataFrame,
    start: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Rebuild daily share counts for the securities sleeve.

    Returns (daily, shares, warnings) where `daily` has columns
      value  : end-of-day market value of holdings
      flow   : net money put into holdings that day (buys − sells ± transfers)
      income : dividends/interest paid by holdings that day (net of ticker fees)
    """
    warnings: list[str] = []
    sec = positions[positions["asset_kind"] != "cash"]
    now_shares = sec.groupby("ticker")["quantity"].sum()

    acts = activities.copy()
    start = pd.Timestamp(start) if start is not None else (
        acts["date"].min() if len(acts) else close.index[-1] - pd.Timedelta(days=365))
    acts = acts[acts["date"] >= start]

    tickers = sorted(set(now_shares.index) | set(acts.loc[acts["type"].isin(_SHARE_SIGN), "ticker"].dropna()))
    no_px = [t for t in tickers if t not in close.columns]
    if no_px:
        warnings.append(f"Excluded (no price history): {', '.join(no_px)}")
    tickers = [t for t in tickers if t in close.columns]

    idx = close.loc[close.index >= start].index
    if len(idx) < 2:
        raise ValueError("Not enough price history in the performance window")
    close = close.loc[idx, tickers]

    # Snap each trade to a trading day (weekend/holiday → next session)
    def snap(d):
        pos = idx.searchsorted(d)
        return idx[min(pos, len(idx) - 1)]

    shares = pd.DataFrame(0.0, index=idx, columns=tickers)
    flow = pd.Series(0.0, index=idx)
    income = pd.Series(0.0, index=idx)
    delta = pd.DataFrame(0.0, index=idx, columns=tickers)

    for _, a in acts.iterrows():
        t, typ = a["ticker"], a["type"]
        if t not in tickers:
            continue
        d = snap(a["date"])
        if typ in _SHARE_SIGN:
            sp = splits[t] if t in splits.columns else pd.Series(dtype=float)
            qty = (a["quantity"] or 0.0) * _split_factor_after(sp, a["date"])
            delta.at[d, t] += _SHARE_SIGN[typ] * qty
            if typ in ("BUY", "REI", "SELL"):
                amt = a["amount"]
                if pd.isna(amt):
                    amt = -_SHARE_SIGN[typ] * (a["quantity"] or 0) * (a["price"] or 0)
                flow[d] += -amt                     # buy: amount<0 → flow in (+)
            elif typ in ("TRANSFER_IN", "TRANSFER_OUT"):
                flow[d] += _SHARE_SIGN[typ] * qty * close.at[d, t]
        elif typ in _INCOME:
            income[d] += a["amount"] or 0.0
        elif typ == "FEE":
            income[d] += a["amount"] or 0.0        # negative amount

    # Backward roll: shares on day d = today's shares − changes strictly after d
    future_changes = delta[::-1].cumsum()[::-1].shift(-1).fillna(0.0)
    for t in tickers:
        shares[t] = now_shares.get(t, 0.0) - future_changes[t]
    neg = shares.lt(-1e-6).any()
    if neg.any():
        warnings.append(
            "History doesn't reconcile to current shares for "
            f"{', '.join(neg[neg].index)} (missing trades or corporate actions) — clipped at 0")
    shares = shares.clip(lower=0.0)

    daily = pd.DataFrame({"value": (shares * close).sum(axis=1), "flow": flow, "income": income})
    return daily, shares, warnings


def daily_twr(daily: pd.DataFrame) -> pd.Series:
    """Daily sleeve returns: (V_t + D_t − F_t) / V_{t−1} − 1 (flows at end of day)."""
    v, f, d = daily["value"], daily["flow"], daily["income"]
    prev = v.shift(1)
    r = (v + d - f) / prev - 1
    first_buy = (prev.fillna(0) <= 0) & (f > 0)
    r[first_buy] = (v[first_buy] + d[first_buy]) / f[first_buy] - 1
    return r.replace([np.inf, -np.inf], np.nan).fillna(0.0).iloc[1:]


def xirr(dates: list, amounts: list) -> float | None:
    """Money-weighted annual return. Negative amounts = money in, positive = money out."""
    if len(amounts) < 2 or not (min(amounts) < 0 < max(amounts)):
        return None
    t0 = min(dates)
    yrs = np.array([(d - t0).days / 365.25 for d in dates])
    amt = np.array(amounts, dtype=float)
    npv = lambda r: float(np.sum(amt / (1 + r) ** yrs))
    try:
        return brentq(npv, -0.9999, 100.0)
    except ValueError:
        return None


def realized_performance(
    daily: pd.DataFrame,
    bench_adj_close: pd.Series,
    risk_free_rate: float = 0.05,
) -> dict:
    r = daily_twr(daily)
    nav = (1 + r).cumprod()
    days = (r.index[-1] - daily.index[0]).days
    years = max(days / 365.25, 1e-9)
    twr = float(nav.iloc[-1] - 1)

    # Money-weighted: start value in, flows in/out, income out, end value out
    cf_dates, cf_amts = [daily.index[0]], [-float(daily["value"].iloc[0])]
    for d, row in daily.iloc[1:].iterrows():
        net = -row["flow"] + row["income"]
        if abs(net) > 1e-9:
            cf_dates.append(d); cf_amts.append(net)
    cf_dates.append(daily.index[-1]); cf_amts.append(float(daily["value"].iloc[-1]))
    mwr = xirr(cf_dates, cf_amts)

    # Benchmark TWR + public-market equivalent (same flows into the benchmark)
    b = bench_adj_close.reindex(daily.index).ffill()
    br = b.pct_change().fillna(0.0)
    bench_twr = float(b.iloc[-1] / b.iloc[0] - 1)
    pme = float(daily["value"].iloc[0])
    for d in daily.index[1:]:
        pme = pme * (1 + br[d]) + daily.at[d, "flow"] - daily.at[d, "income"]

    def period(n_days=None, since=None):
        s = r[r.index >= since] if since is not None else r.tail(n_days)
        return float((1 + s).prod() - 1) if len(s) else np.nan

    last = r.index[-1]
    periods = {
        "1M": period(since=last - pd.DateOffset(months=1)),
        "3M": period(since=last - pd.DateOffset(months=3)),
        "YTD": period(since=pd.Timestamp(last.year, 1, 1)),
        "1Y": period(since=last - pd.DateOffset(years=1)) if years >= 0.99 else np.nan,
        "Since start": twr,
    }
    bnav = (1 + br.iloc[1:]).cumprod()
    return {
        "start": daily.index[0], "end": last, "years": years,
        "twr": twr,
        "twr_ann": (1 + twr) ** (1 / years) - 1 if years >= 1 else np.nan,
        "mwr": mwr,
        "bench_twr": bench_twr,
        "start_value": float(daily["value"].iloc[0]),
        "end_value": float(daily["value"].iloc[-1]),
        "net_flows": float(daily["flow"].iloc[1:].sum()),
        "income": float(daily["income"].iloc[1:].sum()),
        "pme_end_value": pme,
        "pme_excess": float(daily["value"].iloc[-1] - pme),
        "vol": float(r.std() * np.sqrt(252)),
        "max_drawdown": float(compute_max_drawdown(nav)),
        "periods": periods,
        "nav": pd.DataFrame({"Portfolio": nav, "Benchmark": bnav}),
        "drawdown": compute_drawdown_series(nav),
        "daily_returns": r,
    }
