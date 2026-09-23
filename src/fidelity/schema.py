"""
schema.py
---------
Common data shapes shared by every Fidelity data source (SnapTrade API or CSV
export), so the analytics and rebalance code never care where data came from.

positions  : one row per (account, ticker)
activities : one row per transaction
"""

from __future__ import annotations

import re
import pandas as pd

POSITION_COLUMNS = [
    "account_id",      # stable id (SnapTrade UUID or Fidelity account number)
    "account_name",
    "account_type",    # "taxable" | "tax_deferred" | "tax_free"
    "ticker",
    "description",
    "asset_kind",      # "equity" | "etf" | "mutualfund" | "option" | "cash" | "other"
    "quantity",
    "price",
    "market_value",
    "cost_basis",      # total cost basis for the position (NaN if unknown)
    "lots",            # list[dict] of tax lots (may be empty)
]

ACTIVITY_COLUMNS = [
    "account_id",
    "date",            # trade date (Timestamp, normalized to midnight)
    "type",            # BUY | SELL | DIVIDEND | REI | CONTRIBUTION | WITHDRAWAL |
                       # INTEREST | FEE | TRANSFER_IN | TRANSFER_OUT | SPLIT | OTHER
    "ticker",
    "quantity",        # always positive; direction comes from `type`
    "price",
    "amount",          # signed cash impact on the account (+ in / - out)
    "description",
]

# Fidelity core positions / money-market sweep funds — treated as cash.
CASH_TICKERS = {
    "SPAXX", "FDRXX", "FZFXX", "SPRXX", "FZDXX", "FCASH", "CORE", "FMPXX",
    "FTEXX", "FGTXX", "FSIXX", "FZCXX", "FNSXX", "USD", "CUR:USD", "CASH",
}

_TAX_DEFERRED = re.compile(r"\b(IRA|401\s?\(?K\)?|403\s?\(?B\)?|457|SEP|SIMPLE|ROLLOVER|TRADITIONAL|BROKERAGELINK)\b", re.I)
_TAX_FREE = re.compile(r"\b(ROTH|HSA|529)\b", re.I)


def classify_account(name: str | None) -> str:
    """Best-effort tax classification from the account name/type string."""
    name = name or ""
    if _TAX_FREE.search(name):
        return "tax_free"
    if _TAX_DEFERRED.search(name):
        return "tax_deferred"
    return "taxable"


def is_cash_ticker(ticker: str | None) -> bool:
    if not isinstance(ticker, str) or not ticker:
        return False
    t = ticker.upper().rstrip("*").strip()
    return t in CASH_TICKERS


def empty_positions() -> pd.DataFrame:
    return pd.DataFrame(columns=POSITION_COLUMNS)


def empty_activities() -> pd.DataFrame:
    return pd.DataFrame(columns=ACTIVITY_COLUMNS)


def finalize_positions(df: pd.DataFrame, account_types: dict[str, str] | None = None) -> pd.DataFrame:
    """Coerce types, fill derived fields, apply account-type overrides."""
    df = df.copy()
    for col in POSITION_COLUMNS:
        if col not in df.columns:
            df[col] = [[] for _ in range(len(df))] if col == "lots" else pd.NA
    for col in ["quantity", "price", "market_value", "cost_basis"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["ticker"] = df["ticker"].astype(str).str.upper().str.rstrip("*").str.strip()
    cash_mask = df["ticker"].map(is_cash_ticker)
    df.loc[cash_mask, "asset_kind"] = "cash"

    # Fill market value / price from each other where one is missing
    mv_missing = df["market_value"].isna() & df["quantity"].notna() & df["price"].notna()
    df.loc[mv_missing, "market_value"] = df.loc[mv_missing, "quantity"] * df.loc[mv_missing, "price"]
    px_missing = df["price"].isna() & df["quantity"].gt(0) & df["market_value"].notna()
    df.loc[px_missing, "price"] = df.loc[px_missing, "market_value"] / df.loc[px_missing, "quantity"]
    # Cash rows: quantity == dollars, price == 1
    df.loc[cash_mask & df["quantity"].isna(), "quantity"] = df.loc[cash_mask, "market_value"]
    df.loc[cash_mask, "price"] = 1.0

    df["account_type"] = df["account_type"].fillna(df["account_name"].map(classify_account))
    if account_types:
        for acct, typ in account_types.items():
            hit = (df["account_id"].astype(str) == str(acct)) | (df["account_name"] == acct)
            df.loc[hit, "account_type"] = typ

    df["lots"] = df["lots"].map(lambda x: x if isinstance(x, list) else [])
    df = df[df["market_value"].fillna(0).abs() > 0.005]
    return df[POSITION_COLUMNS].reset_index(drop=True)


def finalize_activities(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ACTIVITY_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
    for col in ["quantity", "price", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["quantity"] = df["quantity"].abs()
    df["ticker"] = df["ticker"].map(
        lambda t: None if t is None or (not isinstance(t, str) and pd.isna(t)) or not str(t).strip()
        else str(t).upper().rstrip("*").strip()).astype(object)
    df = df.dropna(subset=["date"]).sort_values("date")
    return df[ACTIVITY_COLUMNS].reset_index(drop=True)
