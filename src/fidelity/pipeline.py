"""
pipeline.py
-----------
End-to-end Fidelity workflow:
  load positions + history (SnapTrade or CSV) → market data → performance
  → rebalance plan → HTML report + CSV exports.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime

import pandas as pd

from src.config import AppConfig
from src.fidelity.schema import empty_activities
from src.fidelity.rebalance import RebalanceSettings, build_rebalance_plan
from src.fidelity import performance as perf

logger = logging.getLogger(__name__)


def resolve_source(cfg: AppConfig) -> str:
    src = cfg.fidelity.source
    if src == "auto":
        src = "snaptrade" if os.getenv("SNAPTRADE_CLIENT_ID") and os.getenv("SNAPTRADE_CONSUMER_KEY") else "csv"
    return src


def load_fidelity_data(cfg: AppConfig, source: str | None = None,
                       positions_csv: str | None = None,
                       history_csv: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    fc = cfg.fidelity
    source = source or resolve_source(cfg)

    if source == "snaptrade":
        from src.fidelity.snaptrade_source import SnapTradeFidelity
        st = SnapTradeFidelity(broker_slug=fc.snaptrade_broker)
        accts = st.accounts()
        if not accts:
            raise RuntimeError("No Fidelity accounts on SnapTrade — run `fidelity-connect` first.")
        positions = st.positions(accts)
        activities = st.activities(accts, start=fc.history_start)
        if fc.account_types:
            from src.fidelity.schema import finalize_positions
            positions = finalize_positions(positions, fc.account_types)
        return positions, activities, "SnapTrade"

    from src.fidelity.csv_source import load_positions_csv, load_activity_csv, latest_file
    import glob
    positions_csv = positions_csv or latest_file(fc.csv_dir, fc.positions_glob)
    if not positions_csv:
        raise FileNotFoundError(
            f"No positions CSV found in {fc.csv_dir}/ (pattern {fc.positions_glob}). "
            "Download it from Fidelity → Positions → Download.")
    positions = load_positions_csv(positions_csv, fc.account_types)
    history_csv = history_csv or sorted(glob.glob(os.path.join(os.path.expanduser(fc.csv_dir), fc.history_glob)))
    activities = load_activity_csv(history_csv) if history_csv else empty_activities()
    if fc.history_start and len(activities):
        activities = activities[activities["date"] >= pd.Timestamp(fc.history_start)]
    return positions, activities, f"CSV ({os.path.basename(positions_csv)})"


def settings_from_config(cfg: AppConfig) -> RebalanceSettings:
    r, o = cfg.fidelity.rebalance, cfg.optimizer
    return RebalanceSettings(
        target_mode=r.target_mode, model_universe=r.model_universe, static_targets=r.static_targets,
        unmanaged=r.unmanaged, accounts=r.accounts, managed_accounts=r.managed_accounts,
        exclude_managed_sleeves=r.exclude_managed_sleeves, max_mvo_assets=r.max_mvo_assets, cash_target_pct=r.cash_target_pct,
        drift_band=r.drift_band, min_trade_usd=r.min_trade_usd, max_turnover=r.max_turnover,
        fractional_shares=r.fractional_shares, avoid_short_term_gains=r.avoid_short_term_gains,
        tlh_loss_pct=r.tlh_loss_pct, tlh_min_usd=r.tlh_min_usd,
        lookback_days=o.lookback_days, weight_min=o.weight_min, weight_max=o.weight_max,
        transaction_cost_bps=o.transaction_cost_bps, risk_aversion=r.risk_aversion,
    )


def run_fidelity(
    cfg: AppConfig,
    source: str | None = None,
    positions_csv: str | None = None,
    history_csv: list[str] | None = None,
    rebalance: bool = True,
    output: str | None = None,
    market_data: dict | None = None,
) -> dict:
    positions, activities, src_label = load_fidelity_data(cfg, source, positions_csv, history_csv)
    fc, bench = cfg.fidelity, cfg.universe.benchmark
    rf = cfg.performance.risk_free_rate
    s = settings_from_config(cfg)
    warnings: list[str] = []

    sec_tickers = positions.loc[positions["asset_kind"].isin(["equity", "etf", "mutualfund"]), "ticker"]
    act_tickers = activities["ticker"].dropna() if len(activities) else pd.Series(dtype=str)
    tickers = set(sec_tickers) | set(act_tickers) | {bench} | set(s.model_universe) | set(s.static_targets)
    skipped = positions.loc[positions["asset_kind"].isin(["option", "fixed_income", "other"]), "ticker"].tolist()
    if skipped:
        warnings.append(f"Options / bonds / other not modelled (kept at current value): {', '.join(skipped)}")

    hist_start = activities["date"].min() if len(activities) else None
    lookback_start = pd.Timestamp.today() - pd.Timedelta(days=int(fc.analytics_lookback_days * 1.5) + 10)
    start = min(filter(None, [hist_start, lookback_start]))
    md = market_data or perf.fetch_market_data(sorted(tickers), start=start)

    holdings = None
    try:
        holdings = perf.holdings_analytics(positions, md["adj_close"], bench, rf, fc.analytics_lookback_days)
        if holdings["unpriced"]:
            warnings.append(f"No price history (excluded from risk profile): {', '.join(holdings['unpriced'])}")
    except Exception as e:
        warnings.append(f"Holdings analytics skipped: {e}")

    realized = None
    if len(activities):
        try:
            daily, _, w = perf.reconstruct_history(positions, activities, md["close"], md["splits"], start=hist_start)
            warnings += w
            realized = perf.realized_performance(daily, md["adj_close"][bench], rf)
        except Exception as e:
            warnings.append(f"Realized performance skipped: {e}")
    else:
        warnings.append("No transaction history loaded — realized TWR / money-weighted return not computed. "
                        "Add Fidelity history CSVs (Activity & Orders → History → Download) or use SnapTrade.")

    plan = None
    if rebalance:
        try:
            plan = build_rebalance_plan(positions, md["adj_close"], s, activities)
        except ValueError as e:
            warnings.append(f"Rebalance skipped: {e}")

    out_dir = fc.output_dir
    stamp = datetime.now().strftime("%Y%m%d")
    output = output or os.path.join(out_dir, f"fidelity_report_{stamp}.html")
    from src.fidelity.report import generate_fidelity_report
    generate_fidelity_report(positions, holdings, realized, plan, warnings, output, bench, src_label)

    os.makedirs(out_dir, exist_ok=True)
    positions.drop(columns=["lots"]).to_csv(os.path.join(out_dir, f"fidelity_positions_{stamp}.csv"), index=False)
    if plan is not None:
        plan["trades"].to_csv(os.path.join(out_dir, f"fidelity_trades_{stamp}.csv"), index=False)

    return {"positions": positions, "activities": activities, "holdings": holdings,
            "realized": realized, "plan": plan, "warnings": warnings, "report": output,
            "source": src_label}
