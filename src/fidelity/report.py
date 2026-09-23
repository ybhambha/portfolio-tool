"""
report.py
---------
Self-contained HTML report for the Fidelity portfolio: holdings, realized
performance, risk, and the proposed rebalance trade list.
"""

from __future__ import annotations

import base64
import html
import os
from datetime import datetime
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, ORANGE, RED, GREY = "#2563eb", "#f59e0b", "#dc2626", "#6b7280"


def _img(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


def _pct(x, d=2):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x*100:,.{d}f}%"


def _usd(x, d=0):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"-${-x:,.{d}f}" if x < 0 else f"${x:,.{d}f}"


def _table(df: pd.DataFrame, fmt: dict | None = None, index=False) -> str:
    fmt = fmt or {}
    d = df.copy()
    for c, f in fmt.items():
        if c in d.columns:
            d[c] = d[c].map(f)
    return d.to_html(index=index, escape=True, border=0, classes="t", na_rep="—")


def _nav_chart(nav: pd.DataFrame, title: str) -> str:
    fig, ax = plt.subplots(figsize=(10, 3.4))
    colors = [BLUE, ORANGE, GREY]
    for i, c in enumerate(nav.columns):
        ax.plot(nav.index, nav[c] * 100 - 100, label=c, color=colors[i % 3], lw=1.6)
    ax.axhline(0, color="#d1d5db", lw=0.8)
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_ylabel("Cumulative return (%)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    return _img(fig)


def _dd_chart(dd: pd.Series) -> str:
    fig, ax = plt.subplots(figsize=(10, 2.2))
    ax.fill_between(dd.index, dd.values * 100, 0, color=RED, alpha=0.35)
    ax.set_title("Drawdown (%)", loc="left", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    return _img(fig)


def _weights_chart(w: pd.DataFrame) -> str:
    w = w[(w[["current", "target", "after"]].abs().sum(axis=1)) > 0.001]
    fig, ax = plt.subplots(figsize=(10, max(2.5, 0.32 * len(w))))
    y = np.arange(len(w))
    ax.barh(y - 0.2, w["current"] * 100, 0.4, label="Current", color=GREY)
    ax.barh(y + 0.2, w["target"] * 100, 0.4, label="Target", color=BLUE)
    ax.set_yticks(y, w.index)
    ax.invert_yaxis()
    ax.set_xlabel("Weight (%)")
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    return _img(fig)


CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#111827}
h1{font-size:24px;margin-bottom:2px} h2{font-size:18px;margin-top:36px;border-bottom:1px solid #e5e7eb;padding-bottom:6px}
.sub{color:#6b7280;font-size:13px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:16px 0}
.card{border:1px solid #e5e7eb;border-radius:8px;padding:12px}
.card .l{font-size:12px;color:#6b7280}.card .v{font-size:20px;font-weight:600;margin-top:4px}
table.t{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}
table.t th{text-align:left;background:#f9fafb;border-bottom:1px solid #e5e7eb;padding:6px}
table.t td{border-bottom:1px solid #f3f4f6;padding:6px}
img{max-width:100%}
.warn{background:#fffbeb;border:1px solid #fde68a;border-radius:8px;padding:10px 14px;font-size:13px}
.note{font-size:12px;color:#6b7280}
"""


def _cards(items) -> str:
    return '<div class="cards">' + "".join(
        f'<div class="card"><div class="l">{html.escape(l)}</div><div class="v">{v}</div></div>'
        for l, v in items) + "</div>"


def generate_fidelity_report(
    positions: pd.DataFrame,
    holdings: dict | None,
    realized: dict | None,
    plan: dict | None,
    warnings: list[str],
    output_path: str,
    benchmark: str = "SPY",
    source: str = "",
) -> str:
    parts = [f"<h1>Fidelity Portfolio Report</h1><div class='sub'>Generated "
             f"{datetime.now():%Y-%m-%d %H:%M} · source: {html.escape(source)}</div>"]

    total = positions["market_value"].sum()
    cash = positions.loc[positions["asset_kind"] == "cash", "market_value"].sum()
    unreal = (positions["market_value"] - positions["cost_basis"])[positions["asset_kind"] != "cash"].sum()
    items = [("Total value", _usd(total)), ("Cash", f"{_usd(cash)} ({_pct(cash/total,1)})"),
             ("Accounts", str(positions["account_id"].nunique())),
             ("Unrealized gain", _usd(unreal))]
    if realized:
        items += [("TWR (since start)", _pct(realized["twr"])),
                  ("Money-weighted (ann.)", _pct(realized["mwr"]))]
    parts.append(_cards(items))

    if warnings:
        parts.append("<div class='warn'><b>Data notes</b><ul>" +
                     "".join(f"<li>{html.escape(w)}</li>" for w in warnings) + "</ul></div>")

    # Holdings
    h = positions.groupby(["ticker", "description", "asset_kind"], dropna=False, as_index=False) \
                 .agg(quantity=("quantity", "sum"), market_value=("market_value", "sum"),
                      cost_basis=("cost_basis", lambda x: x.sum(min_count=len(x))))  # unknown if any lot unknown
    h["weight"] = h["market_value"] / total
    h["gain"] = h["market_value"] - h["cost_basis"]
    h = h.sort_values("market_value", ascending=False)
    parts.append("<h2>Holdings (all accounts)</h2>")
    parts.append(_table(h, {"quantity": lambda x: f"{x:,.3f}", "market_value": _usd,
                            "cost_basis": _usd, "weight": _pct, "gain": _usd}))
    acct = positions.groupby(["account_name", "account_type"], as_index=False)["market_value"].sum()
    parts.append("<h3>By account</h3>" + _table(acct, {"market_value": _usd}))

    # Realized
    if realized:
        parts.append(f"<h2>Realized performance</h2><div class='sub'>{realized['start']:%Y-%m-%d} → "
                     f"{realized['end']:%Y-%m-%d} · invested sleeve (cash excluded), "
                     "reconstructed from transaction history</div>")
        per = pd.DataFrame({"Return": realized["periods"]}).T
        parts.append(_table(per, {c: _pct for c in per.columns}))
        parts.append(_cards([
            ("Time-weighted", _pct(realized["twr"])),
            (f"{benchmark} same period", _pct(realized["bench_twr"])),
            ("Money-weighted (XIRR)", _pct(realized["mwr"])),
            ("Volatility (ann.)", _pct(realized["vol"])),
            ("Max drawdown", _pct(realized["max_drawdown"])),
            (f"vs {benchmark} with same cash flows", _usd(realized["pme_excess"])),
        ]))
        parts.append(f"<p class='note'>Start value {_usd(realized['start_value'])}, net purchases "
                     f"{_usd(realized['net_flows'])}, income {_usd(realized['income'])}, end value "
                     f"{_usd(realized['end_value'])}. Putting the same cash flows into {benchmark} would have "
                     f"grown to {_usd(realized['pme_end_value'])}.</p>")
        nav = realized["nav"].rename(columns={"Benchmark": benchmark})
        parts.append(_nav_chart(nav, "Portfolio (TWR) vs benchmark"))
        parts.append(_dd_chart(realized["drawdown"]))

    # Holdings analytics
    if holdings:
        a, b = holdings["window"]
        parts.append(f"<h2>Risk profile of today's holdings</h2><div class='sub'>Current weights applied to "
                     f"{a:%Y-%m-%d} → {b:%Y-%m-%d} prices (what-if, not your realized return)</div>")
        parts.append(_table(holdings["metrics"].reset_index().rename(columns={"index": "Metric"})))
        parts.append(_cards([("Beta vs " + benchmark, f"{holdings['beta']:.2f}"),
                             ("Tracking error", _pct(holdings["tracking_error"]))]))
        parts.append(_nav_chart(holdings["nav"], "Current holdings, back-tested"))
        parts.append("<h3>Risk contribution</h3>")
        parts.append(_table(holdings["risk"].reset_index().rename(columns={"index": "ticker"}),
                            {"weight": _pct, "ann_vol": _pct, "beta": lambda x: f"{x:.2f}",
                             "risk_contrib": _pct}))

    # Rebalance
    if plan:
        s = plan["summary"]
        parts.append("<h2>Proposed rebalance</h2>")
        parts.append(_cards([
            ("Trades", str(s["n_trades"])), ("Sell", _usd(s["sell_value"])), ("Buy", _usd(s["buy_value"])),
            ("Turnover", _pct(s["turnover"]) + (" (capped)" if s["turnover_capped"] else "")),
            ("Est. realized gain (taxable)", _usd(s["est_realized_gain"])),
            ("of which short-term", _usd(s["est_st_gain"])),
        ]))
        if s["unmanaged"]:
            parts.append(f"<p class='note'>Held outside the model and left untouched: "
                         f"{html.escape(', '.join(s['unmanaged']))} ({_usd(s['unmanaged_value'])}).</p>")
        parts.append(_weights_chart(plan["weights"]))
        parts.append(_table(plan["weights"].reset_index().rename(columns={"index": "ticker"}),
                            {"current": _pct, "target": _pct, "after": _pct}))
        parts.append("<h3>Trade list</h3>")
        if plan["trades"].empty:
            parts.append("<p>No trades needed — everything is within the drift band.</p>")
        else:
            t = plan["trades"].drop(columns=["account_id"])
            parts.append(_table(t, {"est_price": lambda x: _usd(x, 2), "est_value": _usd,
                                    "est_gain": _usd, "st_gain": _usd, "lt_gain": _usd}))
        parts.append("<p class='note'>Estimates use the latest available prices. Review every order before "
                     "placing it in Fidelity — this tool never trades. Tax figures are estimates, not tax advice.</p>")
        if not plan["tlh"].empty:
            parts.append("<h3>Tax-loss harvesting candidates (taxable accounts)</h3>")
            parts.append(_table(plan["tlh"], {"market_value": _usd, "cost_basis": _usd,
                                              "unrealized": _usd, "unrealized_pct": _pct}))

    doc = f"<!doctype html><html><head><meta charset='utf-8'><title>Fidelity Portfolio Report</title>" \
          f"<style>{CSS}</style></head><body>{''.join(parts)}</body></html>"
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(doc)
    return output_path
