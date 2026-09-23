"""
snaptrade_source.py
-------------------
Pulls Fidelity positions, cash and transaction history through SnapTrade
(Fidelity has no public retail API; SnapTrade is an authorized aggregator
that connects through Fidelity's own login page).

One-time setup
  1. Create a free personal key at https://dashboard.snaptrade.com
  2. Put SNAPTRADE_CLIENT_ID / SNAPTRADE_CONSUMER_KEY in .env
  3. python -m src.main fidelity-connect  → open the printed link, log in to Fidelity
  4. python -m src.main fidelity-sync     → positions + history

Personal-key mode is the default (no user id / secret needed). If you use a
commercial key, also set SNAPTRADE_USER_ID and SNAPTRADE_USER_SECRET.

The connection is requested read-only: this tool never places orders.
"""

from __future__ import annotations

import os
import time
import logging
import decimal
from datetime import date

import pandas as pd

from src.fidelity.schema import (
    classify_account, finalize_positions, finalize_activities,
    empty_activities, is_cash_ticker,
)

logger = logging.getLogger(__name__)

_TYPE_MAP = {
    "BUY": "BUY", "SELL": "SELL",
    "DIVIDEND": "DIVIDEND", "SUBSTITUTE_DIVIDEND": "DIVIDEND",
    "REI": "REI",
    "CONTRIBUTION": "CONTRIBUTION", "DEPOSIT": "CONTRIBUTION",
    "WITHDRAWAL": "WITHDRAWAL",
    "INTEREST": "INTEREST",
    "FEE": "FEE", "TAX": "FEE",
    "SPLIT": "SPLIT",
    "STOCK_DIVIDEND": "STOCK_DIV",
    "EXTERNAL_ASSET_TRANSFER_IN": "TRANSFER_IN",
    "EXTERNAL_ASSET_TRANSFER_OUT": "TRANSFER_OUT",
}


def _plain(x):
    """Convert SnapTrade schema objects (frozendict / Decimal / BoolClass …) to plain Python."""
    if x is None:
        return None
    if hasattr(x, "is_none_oapg") and x.is_none_oapg():
        return None
    if hasattr(x, "is_true_oapg"):
        return bool(x.is_true_oapg())
    if isinstance(x, bool):
        return x
    if isinstance(x, decimal.Decimal):
        return float(x)
    if isinstance(x, str):
        return str(x)
    if isinstance(x, dict) or hasattr(x, "items"):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    return x


def _num(x):
    try:
        return float(x) if x is not None and x != "" else None
    except (TypeError, ValueError):
        return None


class SnapTradeFidelity:
    """Thin, read-only wrapper around the SnapTrade SDK for Fidelity accounts."""

    def __init__(
        self,
        client_id: str | None = None,
        consumer_key: str | None = None,
        user_id: str | None = None,
        user_secret: str | None = None,
        broker_slug: str = "FIDELITY",
        institution_filter: str | None = "Fidelity",
        max_retries: int = 5,
    ):
        from snaptrade_client import SnapTrade
        from snaptrade_client.auth import SnapTradeAuth

        client_id = client_id or os.getenv("SNAPTRADE_CLIENT_ID")
        consumer_key = consumer_key or os.getenv("SNAPTRADE_CONSUMER_KEY")
        if not client_id or not consumer_key:
            raise RuntimeError(
                "SnapTrade credentials missing — set SNAPTRADE_CLIENT_ID and "
                "SNAPTRADE_CONSUMER_KEY in .env (see .env.example)."
            )
        self.user_id = user_id or os.getenv("SNAPTRADE_USER_ID")
        self.user_secret = user_secret or os.getenv("SNAPTRADE_USER_SECRET")
        self.commercial = bool(self.user_id and self.user_secret)

        auth = (SnapTradeAuth.commercial_api_key if self.commercial
                else SnapTradeAuth.personal_api_key)(consumer_key=consumer_key, client_id=client_id)
        self.client = SnapTrade(auth=auth)
        self.broker_slug = broker_slug
        self.institution_filter = institution_filter
        self.max_retries = max_retries

    # ------------------------------------------------------------------ utils
    def _user_kwargs(self) -> dict:
        return {"user_id": self.user_id, "user_secret": self.user_secret} if self.commercial else {}

    def _call(self, fn, **kwargs):
        """Call an SDK method with the right credentials and 429 back-off."""
        from snaptrade_client.exceptions import ApiException
        kwargs.update(self._user_kwargs())
        for attempt in range(self.max_retries):
            try:
                return _plain(fn(**kwargs).body)
            except ApiException as e:
                if getattr(e, "status", None) != 429 or attempt == self.max_retries - 1:
                    raise
                retry_after = None
                try:
                    retry_after = float(e.headers.get("Retry-After"))  # type: ignore[union-attr]
                except Exception:
                    pass
                wait = retry_after or min(60, 2 ** attempt * 3)
                logger.warning(f"SnapTrade rate limit hit — sleeping {wait:.0f}s")
                time.sleep(wait)

    # -------------------------------------------------------------- connect
    def connection_link(self, reconnect: str | None = None) -> str:
        """Return the SnapTrade portal URL used to (re)connect Fidelity, read-only."""
        kwargs = {"broker": self.broker_slug, "connection_type": "read"}
        if reconnect:
            kwargs["reconnect"] = reconnect
        body = self._call(self.client.authentication.login_snap_trade_user, **kwargs)
        return body.get("redirectURI") or body.get("redirect_uri") or str(body)

    # -------------------------------------------------------------- accounts
    def accounts(self) -> list[dict]:
        accts = self._call(self.client.account_information.list_user_accounts) or []
        out = []
        for a in accts:
            inst = a.get("institution_name") or ""
            if self.institution_filter and self.institution_filter.lower() not in inst.lower():
                continue
            if a.get("account_category") not in (None, "INVESTMENT"):
                continue
            if a.get("status") in ("closed", "archived"):
                continue
            out.append(a)
        logger.info(f"SnapTrade: {len(out)} Fidelity account(s) found")
        return out

    # ------------------------------------------------------------- positions
    def positions(self, accounts: list[dict] | None = None) -> pd.DataFrame:
        accounts = accounts if accounts is not None else self.accounts()
        rows = []
        for a in accounts:
            acct_id = a["id"]
            name = a.get("name") or a.get("number") or acct_id
            acct_type = classify_account(f"{name} {a.get('raw_type') or ''}")

            body = self._call(self.client.account_information.get_all_account_positions,
                              account_id=acct_id) or {}
            for p in body.get("results", []):
                inst = p.get("instrument") or {}
                kind = inst.get("kind", "other")
                if p.get("cash_equivalent"):
                    continue  # already included in the account's cash balance
                units = _num(p.get("units")) or 0.0
                price = _num(p.get("price"))
                avg_cost = _num(p.get("cost_basis"))
                mult = (_num(inst.get("multiplier")) or 1.0) if kind == "option" else 1.0
                ticker = inst.get("symbol") or inst.get("raw_symbol")
                lots = [
                    {
                        "date": l.get("original_purchase_date"),
                        "quantity": _num(l.get("quantity")),
                        "cost_basis": _num(l.get("cost_basis")),
                        "lot_id": l.get("lot_id"),
                    }
                    for l in (p.get("tax_lots") or [])
                ]
                rows.append({
                    "account_id": acct_id,
                    "account_name": name,
                    "account_type": acct_type,
                    "ticker": ticker,
                    "description": inst.get("description"),
                    "asset_kind": {"stock": "equity", "adr": "equity", "cef": "equity"}.get(kind, kind),
                    "quantity": units,
                    "price": price,
                    "market_value": units * price * mult if price is not None else None,
                    "cost_basis": avg_cost * units * mult if avg_cost is not None else None,
                    "lots": lots,
                })

            # Cash balance (USD)
            balances = self._call(self.client.account_information.get_user_account_balance,
                                  account_id=acct_id) or []
            cash = sum(_num(b.get("cash")) or 0.0 for b in balances
                       if ((b.get("currency") or {}).get("code") or "USD") == "USD")
            if abs(cash) > 0.005:
                rows.append({
                    "account_id": acct_id, "account_name": name, "account_type": acct_type,
                    "ticker": "CASH", "description": "Cash & core position",
                    "asset_kind": "cash", "quantity": cash, "price": 1.0,
                    "market_value": cash, "cost_basis": cash, "lots": [],
                })

        return finalize_positions(pd.DataFrame(rows))

    # ------------------------------------------------------------ activities
    def activities(
        self,
        accounts: list[dict] | None = None,
        start: date | str | None = None,
        end: date | str | None = None,
        page_size: int = 1000,
    ) -> pd.DataFrame:
        accounts = accounts if accounts is not None else self.accounts()
        rows = []
        for a in accounts:
            offset = 0
            while True:
                kwargs = {"account_id": a["id"], "offset": offset, "limit": page_size}
                if start:
                    kwargs["start_date"] = pd.Timestamp(start).date()
                if end:
                    kwargs["end_date"] = pd.Timestamp(end).date()
                body = self._call(self.client.account_information.get_account_activities, **kwargs) or {}
                data = body.get("data", body if isinstance(body, list) else [])
                for act in data:
                    raw_type = (act.get("type") or "").upper()
                    typ = _TYPE_MAP.get(raw_type, "OTHER")
                    units = _num(act.get("units")) or 0.0
                    amount = _num(act.get("amount"))
                    if raw_type == "TRANSFER":
                        sign = units if units else (amount or 0)
                        typ = "TRANSFER_IN" if sign > 0 else "TRANSFER_OUT"
                    sym = (act.get("symbol") or {}).get("symbol")
                    rows.append({
                        "account_id": a["id"],
                        "date": act.get("trade_date") or act.get("settlement_date"),
                        "type": typ,
                        "ticker": sym,
                        "quantity": abs(units),
                        "price": _num(act.get("price")),
                        "amount": amount,
                        "description": act.get("description"),
                    })
                if len(data) < page_size:
                    break
                offset += page_size

        if not rows:
            return empty_activities()
        acts = finalize_activities(pd.DataFrame(rows))
        # Money-market sweep buys/sells are cash movements, not trades
        return acts[~acts["ticker"].map(is_cash_ticker)].reset_index(drop=True)
