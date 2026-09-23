"""
tests/test_fidelity.py
----------------------
Fidelity integration: CSV parsing, SnapTrade mapping (mocked), performance
reconstruction (TWR / XIRR), rebalance logic and an end-to-end report run.
No network access needed — market data is synthetic.
"""

import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.fidelity.csv_source import load_positions_csv, load_activity_csv
from src.fidelity.schema import classify_account, finalize_positions, finalize_activities
from src.fidelity import performance as perf
from src.fidelity.rebalance import RebalanceSettings, build_rebalance_plan

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "fidelity")
POS_CSV = os.path.join(FIX, "Portfolio_Positions_Sep-22-2026.csv")
HIST_CSV = os.path.join(FIX, "Accounts_History.csv")


def synthetic_market(tickers=("XLK", "XLE", "XLV", "SPY"), start="2023-01-02", end="2026-09-22", seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end)
    drift = {"XLK": 0.0006, "XLE": 0.0001, "XLV": 0.0003, "SPY": 0.0004}
    px = {}
    for t in tickers:
        r = drift.get(t, 0.0003) + 0.012 * rng.standard_normal(len(idx))
        px[t] = 100 * np.exp(np.cumsum(r))
    close = pd.DataFrame(px, index=idx)
    return {"adj_close": close.copy(), "close": close.copy(),
            "splits": pd.DataFrame(0.0, index=idx, columns=close.columns)}


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------

class TestCsv:
    def test_positions_parse(self):
        p = load_positions_csv(POS_CSV)
        assert set(p["ticker"]) == {"XLK", "XLE", "XLV", "CASH"}
        # Taxable cash = 2,500 core − 500 pending
        cash = p[(p["ticker"] == "CASH") & (p["account_id"] == "Z11111111")]["market_value"].item()
        assert cash == pytest.approx(2000.0)
        assert p.loc[p["ticker"] == "XLE", "cost_basis"].item() == pytest.approx(20000.0)
        assert p["market_value"].sum() == pytest.approx(61600.0)

    def test_account_types(self):
        p = load_positions_csv(POS_CSV)
        types = p.drop_duplicates("account_id").set_index("account_id")["account_type"]
        assert types["Z11111111"] == "taxable"
        assert types["222222222"] == "tax_free"
        assert classify_account("Rollover IRA") == "tax_deferred"
        assert classify_account("Health Savings Account HSA") == "tax_free"

    def test_account_type_override(self):
        p = load_positions_csv(POS_CSV, {"Z11111111": "tax_deferred"})
        assert (p.loc[p["account_id"] == "Z11111111", "account_type"] == "tax_deferred").all()

    def test_history_parse(self):
        a = load_activity_csv(HIST_CSV)
        assert list(a["type"]) == ["BUY", "SELL", "CONTRIBUTION", "DIVIDEND", "BUY"]
        assert "SPAXX" not in set(a["ticker"].dropna())          # sweep rows dropped
        assert a.loc[a["type"] == "SELL", "quantity"].item() == 10  # stored positive


# ---------------------------------------------------------------------------
# SnapTrade mapping (SDK mocked)
# ---------------------------------------------------------------------------

class TestSnapTrade:
    def _client(self, monkeypatch):
        from src.fidelity import snaptrade_source as ss
        monkeypatch.setenv("SNAPTRADE_CLIENT_ID", "cid")
        monkeypatch.setenv("SNAPTRADE_CONSUMER_KEY", "key")
        st = ss.SnapTradeFidelity()
        ai = SimpleNamespace(
            list_user_accounts=lambda **k: SimpleNamespace(body=[
                {"id": "A1", "name": "Fidelity Roth IRA", "institution_name": "Fidelity",
                 "account_category": "INVESTMENT", "status": "open"},
                {"id": "B1", "name": "Checking", "institution_name": "Chase"},
            ]),
            get_all_account_positions=lambda **k: SimpleNamespace(body={"results": [
                {"instrument": {"kind": "etf", "symbol": "XLK", "description": "Tech"},
                 "units": "10", "price": "250.5", "cost_basis": "200"},
                {"instrument": {"kind": "mutualfund", "symbol": "SPAXX"}, "units": "500",
                 "price": "1", "cash_equivalent": True},
            ], "data_freshness": {}}),
            get_user_account_balance=lambda **k: SimpleNamespace(body=[
                {"currency": {"code": "USD"}, "cash": 500.0}]),
            get_account_activities=lambda **k: SimpleNamespace(body={"data": [
                {"type": "BUY", "symbol": {"symbol": "XLK"}, "units": 5, "price": 240,
                 "amount": -1200, "trade_date": "2026-05-01T00:00:00Z"},
                {"type": "CONTRIBUTION", "amount": 1000, "trade_date": "2026-04-01T00:00:00Z"},
            ], "pagination": {}}),
        )
        st.client = SimpleNamespace(account_information=ai)
        return st

    def test_positions_mapping(self, monkeypatch):
        st = self._client(monkeypatch)
        accts = st.accounts()
        assert [a["id"] for a in accts] == ["A1"]              # non-Fidelity filtered
        p = st.positions(accts)
        xlk = p[p["ticker"] == "XLK"].iloc[0]
        assert xlk["market_value"] == pytest.approx(2505.0)
        assert xlk["cost_basis"] == pytest.approx(2000.0)       # avg cost × units
        assert xlk["account_type"] == "tax_free"
        # cash-equivalent SPAXX not double counted; cash row from balance
        assert p["market_value"].sum() == pytest.approx(3005.0)

    def test_activities_mapping(self, monkeypatch):
        st = self._client(monkeypatch)
        a = st.activities(st.accounts())
        assert list(a["type"]) == ["CONTRIBUTION", "BUY"]
        assert a.iloc[1]["quantity"] == 5

    def test_plain_converts_sdk_types(self):
        import decimal
        from src.fidelity.snaptrade_source import _plain
        out = _plain({"a": decimal.Decimal("1.5"), "b": ("x", decimal.Decimal("2"))})
        assert out == {"a": 1.5, "b": ["x", 2.0]}


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------

def _single_asset(prices, trades, now_shares):
    idx = prices.index
    close = prices.to_frame("AAA")
    splits = pd.DataFrame(0.0, index=idx, columns=["AAA"])
    pos = finalize_positions(pd.DataFrame([{
        "account_id": "1", "account_name": "Individual", "ticker": "AAA", "asset_kind": "equity",
        "quantity": now_shares, "price": prices.iloc[-1], "market_value": now_shares * prices.iloc[-1],
        "cost_basis": np.nan}]))
    acts = finalize_activities(pd.DataFrame(trades)) if trades else finalize_activities(pd.DataFrame(
        [{"account_id": "1", "date": idx[0], "type": "OTHER"}]))
    return pos, acts, close, splits


class TestPerformance:
    def test_twr_ignores_flows(self):
        idx = pd.bdate_range("2025-01-01", periods=250)
        prices = pd.Series(np.linspace(100, 150, 250), index=idx)
        buy_day = idx[100]
        trades = [{"account_id": "1", "date": buy_day, "type": "BUY", "ticker": "AAA",
                   "quantity": 50, "price": prices[buy_day], "amount": -50 * prices[buy_day]}]
        pos, acts, close, splits = _single_asset(prices, trades, 100)
        daily, shares, w = perf.reconstruct_history(pos, acts, close, splits, start=idx[0])
        assert shares["AAA"].iloc[0] == 50 and shares["AAA"].iloc[-1] == 100
        assert not w
        r = perf.realized_performance(daily, prices)
        assert r["twr"] == pytest.approx(0.5, abs=1e-9)          # price went 100 → 150
        assert r["bench_twr"] == pytest.approx(0.5, abs=1e-9)
        assert r["pme_excess"] == pytest.approx(0.0, abs=1e-6)  # identical to benchmark

    def test_dividend_counts_as_return(self):
        idx = pd.bdate_range("2025-01-01", periods=10)
        prices = pd.Series(100.0, index=idx)
        trades = [{"account_id": "1", "date": idx[5], "type": "DIVIDEND", "ticker": "AAA", "amount": 10.0}]
        pos, acts, close, splits = _single_asset(prices, trades, 10)
        daily, _, _ = perf.reconstruct_history(pos, acts, close, splits, start=idx[0])
        r = perf.realized_performance(daily, prices)
        assert r["twr"] == pytest.approx(0.01)                  # $10 on $1,000

    def test_split_adjustment(self):
        idx = pd.bdate_range("2025-01-01", periods=20)
        prices = pd.Series(50.0, index=idx)                     # split-adjusted series
        splits = pd.DataFrame(0.0, index=idx, columns=["AAA"])
        splits.loc[idx[10], "AAA"] = 2.0
        trades = [{"account_id": "1", "date": idx[5], "type": "BUY", "ticker": "AAA",
                   "quantity": 10, "price": 100, "amount": -1000}]  # pre-split units
        pos, acts, close, _ = _single_asset(prices, trades, 20)
        daily, shares, w = perf.reconstruct_history(pos, acts, close, splits, start=idx[0])
        assert shares["AAA"].iloc[0] == pytest.approx(0.0)
        assert not w
        assert perf.realized_performance(daily, prices)["twr"] == pytest.approx(0.0, abs=1e-9)

    def test_unreconciled_history_warns(self):
        idx = pd.bdate_range("2025-01-01", periods=20)
        prices = pd.Series(100.0, index=idx)
        trades = [{"account_id": "1", "date": idx[5], "type": "BUY", "ticker": "AAA",
                   "quantity": 50, "price": 100, "amount": -5000}]
        pos, acts, close, splits = _single_asset(prices, trades, 10)   # holds fewer than bought
        _, shares, w = perf.reconstruct_history(pos, acts, close, splits, start=idx[0])
        assert w and (shares >= 0).all().all()

    def test_xirr(self):
        d0 = pd.Timestamp("2024-01-01")
        r = perf.xirr([d0, d0 + pd.Timedelta(days=365)], [-1000, 1100])
        assert r == pytest.approx(0.0998, abs=1e-3)             # 365 / 365.25 days
        assert perf.xirr([d0], [100]) is None

    def test_holdings_analytics(self):
        md = synthetic_market()
        pos = load_positions_csv(POS_CSV)
        h = perf.holdings_analytics(pos, md["adj_close"], "SPY", 0.04, 504)
        assert h["risk"]["risk_contrib"].sum() == pytest.approx(1.0)
        assert "Portfolio" in h["metrics"].columns


# ---------------------------------------------------------------------------
# Rebalance
# ---------------------------------------------------------------------------

class TestRebalance:
    def setup_method(self):
        self.pos = load_positions_csv(POS_CSV)
        self.md = synthetic_market()

    def test_static_targets_move_toward_target(self):
        s = RebalanceSettings(target_mode="static", static_targets={"XLK": 0.4, "XLE": 0.3, "XLV": 0.3},
                              max_turnover=1.0, drift_band=0.0)
        plan = build_rebalance_plan(self.pos, self.md["adj_close"], s)
        t = plan["trades"]
        assert set(t.loc[t["action"] == "SELL", "ticker"]) == {"XLK"}
        assert set(t.loc[t["action"] == "BUY", "ticker"]) >= {"XLV"}
        w = plan["weights"]
        before = (w["current"] - w["target"]).abs().drop("CASH").sum()
        after = (w["after"] - w["target"]).abs().drop("CASH").sum()
        assert after < before * 0.25

    def test_sells_prefer_tax_advantaged(self):
        s = RebalanceSettings(target_mode="static", static_targets={"XLK": 0.2, "XLE": 0.4, "XLV": 0.4},
                              max_turnover=1.0, drift_band=0.0)
        sells = build_rebalance_plan(self.pos, self.md["adj_close"], s)["trades"]
        xlk = sells[(sells["action"] == "SELL") & (sells["ticker"] == "XLK")]
        # Roth's 40 sh go first, before any taxable (gain) lots
        assert xlk.iloc[0]["account_type"] == "tax_free" or \
            xlk.loc[xlk["account_type"] == "tax_free", "shares"].sum() == 40

    def test_no_overspend_and_whole_shares(self):
        s = RebalanceSettings(target_mode="static", static_targets={"XLK": 0.2, "XLE": 0.2, "XLV": 0.6},
                              max_turnover=1.0, drift_band=0.0)
        plan = build_rebalance_plan(self.pos, self.md["adj_close"], s)
        assert plan["summary"]["cash_after"] >= -1e-6
        assert (plan["trades"]["shares"] % 1 == 0).all()

    def test_drift_band_suppresses_small_trades(self):
        w = self.pos.groupby("ticker")["market_value"].sum()
        inv = w.drop("CASH").sum()
        targets = (w.drop("CASH") / inv).to_dict()
        s = RebalanceSettings(target_mode="static", static_targets=targets, drift_band=0.05,
                              cash_target_pct=w["CASH"] / w.sum())
        assert build_rebalance_plan(self.pos, self.md["adj_close"], s)["trades"].empty

    def test_turnover_cap(self):
        s = RebalanceSettings(target_mode="static", static_targets={"XLV": 1.0}, max_turnover=0.10,
                              drift_band=0.0)
        plan = build_rebalance_plan(self.pos, self.md["adj_close"], s)
        assert plan["summary"]["turnover"] <= 0.10 + 0.01
        assert plan["summary"]["turnover_capped"]

    def test_mvo_targets_respect_bounds(self):
        s = RebalanceSettings(target_mode="mvo", weight_max=0.5, max_turnover=1.0)
        plan = build_rebalance_plan(self.pos, self.md["adj_close"], s)
        assert plan["targets"].sum() == pytest.approx(1.0)
        assert plan["targets"].max() <= 0.5 + 1e-6

    def test_unmanaged_held(self):
        s = RebalanceSettings(target_mode="static", static_targets={"XLK": 0.5, "XLE": 0.5},
                              unmanaged="hold", drift_band=0.0, max_turnover=1.0)
        plan = build_rebalance_plan(self.pos, self.md["adj_close"], s)
        assert "XLV" not in set(plan["trades"]["ticker"])
        assert plan["summary"]["unmanaged"] == ["XLV"]

    def test_wash_sale_flag_and_tlh(self):
        acts = load_activity_csv(HIST_CSV)       # XLE bought 2026-09-15 in taxable
        s = RebalanceSettings(target_mode="static", static_targets={"XLK": 0.8, "XLV": 0.2},
                              drift_band=0.0, max_turnover=1.0, unmanaged="sell")
        plan = build_rebalance_plan(self.pos, self.md["adj_close"], s, acts,
                                    today=pd.Timestamp("2026-09-22"))
        xle = plan["trades"][(plan["trades"]["ticker"] == "XLE") & (plan["trades"]["action"] == "SELL")]
        assert xle["note"].str.contains("WASH SALE").any()
        tlh = plan["tlh"]
        assert list(tlh["ticker"]) == ["XLE"] and tlh["wash_sale_risk"].all()

    def test_short_term_lots_held_back(self):
        pos = self.pos.copy()
        i = pos.index[(pos["ticker"] == "XLK") & (pos["account_type"] == "taxable")][0]
        pos.at[i, "lots"] = [
            {"date": "2026-08-01", "quantity": 60, "cost_basis": 60 * 200},   # young, at a gain
            {"date": "2020-01-01", "quantity": 40, "cost_basis": 40 * 100},
        ]
        pos.loc[pos["account_type"] == "tax_free", "account_type"] = "taxable"  # force taxable sells
        s = RebalanceSettings(target_mode="static", static_targets={"XLK": 0.1, "XLE": 0.45, "XLV": 0.45},
                              drift_band=0.0, max_turnover=1.0)
        t = build_rebalance_plan(pos, self.md["adj_close"], s, today=pd.Timestamp("2026-09-22"))["trades"]
        z = t[(t["ticker"] == "XLK") & (t["account_id"] == "Z11111111")]
        assert z["shares"].sum() <= 40
        assert z["note"].str.contains("short-term").any()


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

def test_end_to_end_report(tmp_path):
    from src.config import load_config
    from src.fidelity.pipeline import run_fidelity
    cfg = load_config("config.yaml")
    cfg.fidelity.output_dir = str(tmp_path)
    out = tmp_path / "report.html"
    res = run_fidelity(cfg, source="csv", positions_csv=POS_CSV, history_csv=[HIST_CSV],
                       output=str(out), market_data=synthetic_market())
    assert out.exists() and out.stat().st_size > 10_000
    assert res["realized"] is not None and res["holdings"] is not None
    assert res["plan"] is not None
    html = out.read_text()
    for section in ["Holdings (all accounts)", "Realized performance", "Proposed rebalance"]:
        assert section in html


# ---------------------------------------------------------------------------
# Real-export quirks
# ---------------------------------------------------------------------------

def test_history_dedupe_keeps_real_repeats(tmp_path):
    hdr = ("\n\nRun Date,Account,Account Number,Action,Symbol,Description,Type,Price ($),Quantity,"
           "Commission ($),Fees ($),Accrued Interest ($),Amount ($),Settlement Date\n")
    row = '06/30/2026,ROTH IRA,Y1,YOU BOUGHT X (XLK) (Cash),XLK,X,Cash,100,1,"","","",-100,""\n'
    other = '06/29/2026,ROTH IRA,Y1,YOU SOLD X (XLE) (Cash),XLE,X,Cash,50,-2,"","","",100,""\n'
    f1 = tmp_path / "Accounts_History.csv"; f1.write_text(hdr + row + row + other)     # 2 genuine fills
    f2 = tmp_path / "Accounts_History (1).csv"; f2.write_text(hdr + row + row)          # overlap
    a = load_activity_csv([str(f1), str(f2)])
    assert (a["ticker"] == "XLK").sum() == 2 and (a["ticker"] == "XLE").sum() == 1


def test_corporate_action_share_moves(tmp_path):
    hdr = ("Run Date,Account,Account Number,Action,Symbol,Description,Type,Price ($),Quantity,"
           "Commission ($),Fees ($),Accrued Interest ($),Amount ($),Settlement Date\n")
    rows = ('05/01/2026,Joint,Z1,MERGER MER FROM 123#REOR M005 OLDCO (OLD),OLD,OLDCO,Cash,"",-10,"","","","",""\n'
            '05/01/2026,Joint,Z1,MERGER MER PAYOUT #REOR M005 NEWCO (NEW),NEW,NEWCO,Cash,"",5,"","","","",""\n')
    f = tmp_path / "Accounts_History.csv"; f.write_text(hdr + rows)
    a = load_activity_csv(str(f))
    assert dict(zip(a["ticker"], a["type"])) == {"OLD": "TRANSFER_OUT", "NEW": "TRANSFER_IN"}


def test_managed_sleeve_accounts_excluded():
    pos = load_positions_csv(POS_CSV)
    pos.loc[pos["account_id"] == "222222222", "sleeve"] = None
    pos.loc[(pos["account_id"] == "Z11111111") & (pos["ticker"] == "XLE"), "sleeve"] = \
        "Strategic Advisers Tax-Managed US Large Cap SMA"
    s = RebalanceSettings(target_mode="static", static_targets={"XLK": 0.5, "XLV": 0.5},
                          drift_band=0.0, max_turnover=1.0)
    plan = build_rebalance_plan(pos, synthetic_market()["adj_close"], s)
    assert set(plan["trades"]["account_id"]) <= {"222222222"}
    assert plan["summary"]["excluded_managed_value"] == pytest.approx(45000.0)


def test_mvo_refuses_huge_universe():
    s = RebalanceSettings(target_mode="mvo", max_mvo_assets=2)
    with pytest.raises(ValueError, match="too many"):
        build_rebalance_plan(load_positions_csv(POS_CSV), synthetic_market()["adj_close"], s)


def test_short_history_holdings_are_held_not_sold():
    md = synthetic_market()
    adj = md["adj_close"].copy()
    adj.loc[adj.index[:-60], "XLV"] = np.nan          # XLV only ~3 months old
    s = RebalanceSettings(target_mode="mvo", weight_max=0.6, drift_band=0.0, max_turnover=1.0)
    plan = build_rebalance_plan(load_positions_csv(POS_CSV), adj, s)
    assert "XLV" not in set(plan["trades"]["ticker"])
    assert plan["summary"]["short_history"] == ["XLV"]
