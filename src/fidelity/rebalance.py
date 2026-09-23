"""
rebalance.py
------------
Turns target weights into an account-level trade list for the Fidelity
household. Nothing is sent to Fidelity — you review and place the orders.

Steps
  1. Targets      : MVO (existing optimizer, Ledoit-Wolf) or static weights
  2. Sizing       : cash buffer, drift band, minimum trade, turnover cap
  3. Placement    : sells come from tax-advantaged accounts first, then
                    taxable lots with losses, then the smallest gains;
                    buys go where the cash is
  4. Tax checks   : estimated realized gain (short/long term where lot dates
                    are known), optional short-term-gain avoidance, wash-sale
                    flags, and tax-loss-harvest candidates
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.optimizer.covariance import compute_covariance, compute_expected_returns
from src.optimizer.optimizer import optimize_portfolio

logger = logging.getLogger(__name__)

TAX_ADVANTAGED = {"tax_deferred", "tax_free"}


def _nz(x) -> float:
    return 0.0 if x is None or pd.isna(x) else float(x)


@dataclass
class RebalanceSettings:
    target_mode: str = "mvo"                    # "mvo" | "static"
    model_universe: list[str] = field(default_factory=list)   # empty → current holdings
    static_targets: dict[str, float] = field(default_factory=dict)
    unmanaged: str = "hold"                     # "hold" | "sell" — holdings outside the model
    accounts: list[str] = field(default_factory=list)          # empty → all accounts
    managed_accounts: list[str] = field(default_factory=list)  # advisor-run accounts to leave alone
    exclude_managed_sleeves: bool = True        # skip Fidelity SMA sleeves (Strategic Advisers etc.)
    max_mvo_assets: int = 40
    cash_target_pct: float = 0.02
    drift_band: float = 0.02                    # absolute weight drift before trading
    min_trade_usd: float = 100.0
    max_turnover: float = 0.30                  # one-way, fraction of managed value
    fractional_shares: bool = False
    avoid_short_term_gains: bool = True
    tlh_loss_pct: float = 0.05
    tlh_min_usd: float = 250.0
    # optimizer
    lookback_days: int = 252
    weight_min: float = 0.0
    weight_max: float = 0.30
    transaction_cost_bps: float = 5.0
    risk_aversion: float = 1.0


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

def compute_targets(
    s: RebalanceSettings,
    universe: list[str],
    adj_close: pd.DataFrame,
    current_w: pd.Series,
) -> pd.Series:
    if s.target_mode == "static":
        t = pd.Series(s.static_targets, dtype=float)
        if t.sum() <= 0:
            raise ValueError("static_targets is empty")
        return t / t.sum()

    px = adj_close[[u for u in universe if u in adj_close.columns]].tail(s.lookback_days + 1)
    px = px.dropna(axis=1, thresh=int(0.8 * len(px)))
    dropped = sorted(set(universe) - set(px.columns))
    if dropped:
        logger.warning(f"Not enough price history to optimize: {dropped} — excluded from targets")
    log_r = np.log(px / px.shift(1)).iloc[1:].fillna(0.0)
    n = log_r.shape[1]
    if n == 0:
        raise ValueError("No assets with enough history to optimize")
    w_max = max(s.weight_max, 1.0 / n + 1e-6)   # keep problem feasible for small universes
    if w_max != s.weight_max:
        logger.warning(f"weight_max raised to {w_max:.2%} so {n} assets can sum to 100%")
    return optimize_portfolio(
        mu=compute_expected_returns(log_r),
        cov=compute_covariance(log_r, method="ledoit_wolf"),
        current_weights=current_w.reindex(log_r.columns).fillna(0.0),
        weight_min=s.weight_min,
        weight_max=w_max,
        transaction_cost_bps=s.transaction_cost_bps,
        risk_aversion=s.risk_aversion,
    )


# ---------------------------------------------------------------------------
# Tax helpers
# ---------------------------------------------------------------------------

def _sell_tax(pos: pd.Series, qty: float, price: float, today: pd.Timestamp) -> dict:
    """Estimate realized gain for selling `qty` of one account position (HIFO if lots known)."""
    if pos["account_type"] in TAX_ADVANTAGED:
        return {"gain": 0.0, "st_gain": 0.0, "lt_gain": 0.0, "term": "n/a (tax-advantaged)", "st_blocked_qty": 0.0}
    lots = [l for l in (pos["lots"] or []) if l.get("quantity") and l.get("cost_basis") is not None]
    if lots:
        # Highest cost first minimizes gains
        lots = sorted(lots, key=lambda l: l["cost_basis"] / l["quantity"], reverse=True)
        remaining, st, lt = qty, 0.0, 0.0
        for l in lots:
            take = min(remaining, l["quantity"])
            if take <= 0:
                break
            g = take * (price - l["cost_basis"] / l["quantity"])
            held = today - pd.Timestamp(l["date"]).tz_localize(None) if l.get("date") else None
            if held is not None and held < pd.Timedelta(days=365):
                st += g
            else:
                lt += g
            remaining -= take
        return {"gain": st + lt, "st_gain": st, "lt_gain": lt,
                "term": "ST+LT" if st and lt else ("ST" if st else "LT"), "st_blocked_qty": 0.0}
    if pd.notna(pos["cost_basis"]) and pos["quantity"] > 0:
        g = qty * (price - pos["cost_basis"] / pos["quantity"])
        return {"gain": g, "st_gain": np.nan, "lt_gain": np.nan, "term": "unknown (avg cost)", "st_blocked_qty": 0.0}
    return {"gain": np.nan, "st_gain": np.nan, "lt_gain": np.nan, "term": "unknown", "st_blocked_qty": 0.0}


def _short_term_gain_qty(pos: pd.Series, price: float, today: pd.Timestamp) -> float:
    """Shares in lots that are both <1 year old and at a gain."""
    q = 0.0
    for l in pos["lots"] or []:
        if not (l.get("quantity") and l.get("cost_basis") is not None and l.get("date")):
            continue
        young = today - pd.Timestamp(l["date"]).tz_localize(None) < pd.Timedelta(days=365)
        if young and price > l["cost_basis"] / l["quantity"]:
            q += l["quantity"]
    return q


def recent_buys(activities: pd.DataFrame | None, today: pd.Timestamp, days: int = 30) -> set[str]:
    if activities is None or activities.empty:
        return set()
    a = activities[(activities["type"].isin(["BUY", "REI"])) &
                   (activities["date"] >= today - pd.Timedelta(days=days))]
    return set(a["ticker"].dropna())


def tax_loss_candidates(positions: pd.DataFrame, s: RebalanceSettings,
                        activities: pd.DataFrame | None = None,
                        today: pd.Timestamp | None = None) -> pd.DataFrame:
    today = today or pd.Timestamp.today().normalize()
    p = positions[(positions["account_type"] == "taxable") & (positions["asset_kind"] != "cash")
                  & positions["cost_basis"].notna()].copy()
    p["unrealized"] = p["market_value"] - p["cost_basis"]
    p["unrealized_pct"] = p["unrealized"] / p["cost_basis"]
    c = p[(p["unrealized_pct"] <= -s.tlh_loss_pct) & (p["unrealized"] <= -s.tlh_min_usd)].copy()
    rb = recent_buys(activities, today)
    c["wash_sale_risk"] = c["ticker"].isin(rb)
    return c[["account_name", "ticker", "quantity", "market_value", "cost_basis",
              "unrealized", "unrealized_pct", "wash_sale_risk"]].sort_values("unrealized")


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def build_rebalance_plan(
    positions: pd.DataFrame,
    adj_close: pd.DataFrame,
    s: RebalanceSettings,
    activities: pd.DataFrame | None = None,
    today: pd.Timestamp | None = None,
) -> dict:
    today = today or pd.Timestamp.today().normalize()
    pos = positions.copy()
    if s.accounts:
        pos = pos[pos["account_id"].astype(str).isin(s.accounts) | pos["account_name"].isin(s.accounts)]
    managed_mask = pos["account_id"].astype(str).isin(s.managed_accounts) | pos["account_name"].isin(s.managed_accounts)
    if s.exclude_managed_sleeves and "sleeve" in pos.columns:
        # An account with any Strategic Advisers / SMA sleeve is advisor-run: leave all of it (cash too)
        sma = pos["sleeve"].fillna("").str.contains(r"Strategic Advisers|\bSMA\b", case=False)
        managed_mask |= pos["account_id"].isin(pos.loc[sma, "account_id"].unique())
    excluded_value = float(pos.loc[managed_mask, "market_value"].sum())
    pos = pos[~managed_mask]
    if pos.empty:
        raise ValueError("No self-directed positions in the selected accounts")

    # Latest price per ticker (broker price, else last close)
    last_close = adj_close.ffill().iloc[-1] if len(adj_close) else pd.Series(dtype=float)
    price = pos[pos["asset_kind"] != "cash"].groupby("ticker")["price"].last()
    price = price.fillna(last_close.reindex(price.index))

    sec = pos[pos["asset_kind"] != "cash"]
    cash_total = float(pos.loc[pos["asset_kind"] == "cash", "market_value"].sum())
    held = sec.groupby("ticker")["market_value"].sum()

    universe = list(s.static_targets) if s.target_mode == "static" else (
        s.model_universe or [t for t in held.index if t in adj_close.columns])
    if s.target_mode == "mvo" and len(universe) > s.max_mvo_assets:
        raise ValueError(
            f"{len(universe)} holdings is too many to optimize with MVO (limit {s.max_mvo_assets}). "
            "Set fidelity.rebalance.model_universe (e.g. your ETF list), use static_targets, "
            "or restrict fidelity.rebalance.accounts / managed_accounts.")
    unmanaged = [t for t in held.index if t not in universe]
    unmanaged_value = float(held[unmanaged].sum()) if s.unmanaged == "hold" else 0.0

    total_value = float(held.sum()) + cash_total
    managed_value = total_value - unmanaged_value
    investable = managed_value * (1 - s.cash_target_pct)
    managed_held = held.drop(unmanaged) if s.unmanaged == "hold" else held
    current_w = (managed_held / investable) if investable > 0 else managed_held * 0

    targets = compute_targets(s, universe, adj_close, current_w)
    all_t = sorted(set(targets.index) | set(managed_held.index))
    targets = targets.reindex(all_t).fillna(0.0)
    cur = managed_held.reindex(all_t).fillna(0.0)

    # Prices for new buys
    for t in all_t:
        if t not in price.index or pd.isna(price.get(t)):
            price[t] = last_close.get(t, np.nan)
    unpriceable = [t for t in all_t if pd.isna(price.get(t))]
    if unpriceable:
        logger.warning(f"No price for {unpriceable} — skipped")
        targets = targets.drop(unpriceable, errors="ignore")
        all_t = [t for t in all_t if t not in unpriceable]
        cur = cur.reindex(all_t)

    tgt_dollars = targets.reindex(all_t) * investable
    diff = tgt_dollars - cur
    drift = (cur / investable - targets.reindex(all_t)).abs() if investable > 0 else diff * 0
    in_band = (drift < s.drift_band) & (targets.reindex(all_t) > 0)
    diff[in_band] = 0.0
    diff[diff.abs() < s.min_trade_usd] = 0.0

    # Turnover cap (one-way)
    turnover = diff.abs().sum() / 2 / managed_value if managed_value > 0 else 0.0
    scale = 1.0
    if s.max_turnover and turnover > s.max_turnover:
        scale = s.max_turnover / turnover
        diff *= scale
        logger.info(f"Turnover {turnover:.1%} capped to {s.max_turnover:.1%}")

    # ----------------------------------------------------------- sells
    rb = recent_buys(activities, today)
    trades = []
    acct_cash = pos[pos["asset_kind"] == "cash"].groupby("account_id")["market_value"].sum().to_dict()
    for a in pos["account_id"].unique():
        acct_cash.setdefault(a, 0.0)

    sells = diff[diff < 0]   # with unmanaged="sell", non-model holdings have target 0 here
    loss_sold: set[str] = set()
    for t, dollars in sells.items():
        px = float(price[t])
        need_qty = -dollars / px
        full_exit = targets.get(t, 0.0) == 0.0
        rows = sec[sec["ticker"] == t].copy()
        rows["gps"] = np.where(rows["cost_basis"].notna() & (rows["quantity"] > 0),
                               px - rows["cost_basis"] / rows["quantity"], 0.0)
        rows["pri"] = np.where(rows["account_type"].isin(TAX_ADVANTAGED), -1e12, rows["gps"])
        for _, r in rows.sort_values("pri").iterrows():
            if need_qty <= 1e-9:
                break
            qty = min(need_qty, r["quantity"])
            note = []
            if s.avoid_short_term_gains and r["account_type"] == "taxable" and not full_exit:
                blocked = _short_term_gain_qty(r, px, today)
                avail = max(r["quantity"] - blocked, 0.0)
                if qty > avail:
                    note.append(f"held back {qty - avail:.2f} sh in short-term gain lots")
                    qty = avail
            if not s.fractional_shares and not (full_exit and qty >= r["quantity"] - 1e-9):
                qty = math.floor(qty)
            if qty <= 0 or qty * px < s.min_trade_usd and not full_exit:
                continue
            tax = _sell_tax(r, qty, px, today)
            if r["account_type"] == "taxable" and pd.notna(tax["gain"]) and tax["gain"] < 0:
                loss_sold.add(t)
            trades.append({
                "account_id": r["account_id"], "account_name": r["account_name"],
                "account_type": r["account_type"], "ticker": t, "action": "SELL",
                "shares": round(qty, 4), "est_price": px, "est_value": -qty * px,
                "est_gain": tax["gain"], "st_gain": tax["st_gain"], "lt_gain": tax["lt_gain"],
                "tax_term": tax["term"], "note": "; ".join(note),
            })
            acct_cash[r["account_id"]] += qty * px
            need_qty -= qty

    # ------------------------------------------------------------ buys
    # Keep each account's share of the cash buffer
    acct_total = pos.groupby("account_id")["market_value"].sum()
    reserve = {a: acct_total.get(a, 0.0) * s.cash_target_pct for a in acct_cash}
    spendable = {a: max(acct_cash[a] - reserve[a], 0.0) for a in acct_cash}
    buys = diff[diff > 0].sort_values(ascending=False)
    budget = sum(spendable.values())
    if buys.sum() > budget and buys.sum() > 0:
        buys *= budget / buys.sum()
    names = pos.drop_duplicates("account_id").set_index("account_id")
    for t, dollars in buys.items():
        px = float(price[t])
        holders = set(sec.loc[sec["ticker"] == t, "account_id"])
        order = sorted(spendable, key=lambda a: (a not in holders, -spendable[a]))
        remaining = dollars
        for a in order:
            if remaining < s.min_trade_usd or spendable[a] < s.min_trade_usd:
                continue
            spend = min(remaining, spendable[a])
            qty = spend / px if s.fractional_shares else math.floor(spend / px)
            if qty <= 0 or qty * px < s.min_trade_usd:
                continue
            note = []
            if t in loss_sold:
                note.append("WASH SALE: same ticker sold at a loss in this plan")
            trades.append({
                "account_id": a, "account_name": names.at[a, "account_name"],
                "account_type": names.at[a, "account_type"], "ticker": t, "action": "BUY",
                "shares": round(qty, 4), "est_price": px, "est_value": qty * px,
                "est_gain": np.nan, "st_gain": np.nan, "lt_gain": np.nan,
                "tax_term": "", "note": "; ".join(note),
            })
            spendable[a] -= qty * px
            remaining -= qty * px

    for tr in trades:
        if tr["action"] == "SELL" and tr["account_type"] == "taxable" and \
                pd.notna(tr["est_gain"]) and tr["est_gain"] < 0 and tr["ticker"] in rb:
            tr["note"] = "; ".join(filter(None, [tr["note"], "WASH SALE: bought within last 30 days"]))

    trade_df = pd.DataFrame(trades, columns=[
        "account_id", "account_name", "account_type", "ticker", "action", "shares",
        "est_price", "est_value", "est_gain", "st_gain", "lt_gain", "tax_term", "note"])
    trade_df = trade_df.sort_values(["action", "account_name", "ticker"], ascending=[False, True, True]) \
                       .reset_index(drop=True)

    # Post-trade weights
    after = held.copy()
    for _, tr in trade_df.iterrows():
        after[tr["ticker"]] = after.get(tr["ticker"], 0.0) + tr["est_value"]
    cash_after = total_value - after.sum()
    weights = pd.DataFrame({
        "current": held.reindex(sorted(set(held.index) | set(targets.index))).fillna(0) / total_value,
        "target": (targets * investable / total_value),
        "after": after / total_value,
    }).fillna(0.0)
    for t in (unmanaged if s.unmanaged == "hold" else []):
        weights.at[t, "target"] = held[t] / total_value     # held outside the model
    weights.loc["CASH"] = [cash_total / total_value, (managed_value - investable) / total_value,
                           cash_after / total_value]
    weights = weights.sort_values("target", ascending=False)

    taxable_sells = trade_df[(trade_df["action"] == "SELL") & (trade_df["account_type"] == "taxable")]
    summary = {
        "total_value": total_value,
        "managed_value": managed_value,
        "unmanaged": unmanaged if s.unmanaged == "hold" else [],
        "unmanaged_value": unmanaged_value,
        "excluded_managed_value": excluded_value,
        "n_trades": len(trade_df),
        "buy_value": float(trade_df.loc[trade_df["action"] == "BUY", "est_value"].sum()),
        "sell_value": float(-trade_df.loc[trade_df["action"] == "SELL", "est_value"].sum()),
        "turnover": float(trade_df["est_value"].abs().sum() / 2 / managed_value) if managed_value else 0.0,
        "turnover_capped": scale < 1.0,
        "est_cost": float(trade_df["est_value"].abs().sum() * s.transaction_cost_bps / 10_000),
        "est_realized_gain": _nz(taxable_sells["est_gain"].sum()),
        # NaN when no lot dates are available (term unknown), not a misleading $0
        "est_st_gain": float(taxable_sells["st_gain"].sum(min_count=1)) if len(taxable_sells) else 0.0,
        "cash_after": cash_after,
    }
    return {
        "trades": trade_df,
        "weights": weights,
        "targets": targets,
        "summary": summary,
        "tlh": tax_loss_candidates(pos, s, activities, today),
    }
