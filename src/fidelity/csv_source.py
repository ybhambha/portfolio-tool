"""
csv_source.py
-------------
Loads Fidelity's own CSV downloads — no third party involved.

  Positions : Fidelity.com → Accounts → Positions → Download (Portfolio_Positions_*.csv)
  History   : Accounts → Activity & Orders → History → Download (Accounts_History*.csv)

Fidelity's exports carry disclaimer text, "$1,234.56"-style numbers, "--"
placeholders and a trailing comma on each row; this module handles all of it.
"""

from __future__ import annotations

import io
import re
import glob
import logging
import os

import pandas as pd

from src.fidelity.schema import (
    finalize_positions, finalize_activities, empty_activities, is_cash_ticker,
)

logger = logging.getLogger(__name__)


def _money(s: pd.Series) -> pd.Series:
    cleaned = (
        s.astype(str)
        .str.replace(r"[\$,%+\s]", "", regex=True)
        .str.replace(r"^\((.*)\)$", r"-\1", regex=True)   # (123.45) → -123.45
        .replace({"--": None, "n/a": None, "nan": None, "": None, "None": None})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _read_csv_from_header(path: str, header_pattern: str) -> pd.DataFrame:
    """Find the header line (Fidelity sometimes prepends blank/title lines) and parse from there."""
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        lines = f.read().splitlines()
    start = next((i for i, l in enumerate(lines) if re.search(header_pattern, l, re.I)), None)
    if start is None:
        raise ValueError(f"{os.path.basename(path)}: no header line matching '{header_pattern}'")
    body = []
    for l in lines[start:]:
        if not l.strip():
            if body:          # first blank line after data = start of disclaimer block
                break
            continue
        body.append(l)
    df = pd.read_csv(io.StringIO("\n".join(body)), index_col=False, dtype=str,
                     on_bad_lines="skip", skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    return df


def latest_file(folder: str, pattern: str) -> str | None:
    files = glob.glob(os.path.join(os.path.expanduser(folder), pattern))
    return max(files, key=os.path.getmtime) if files else None


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------

def load_positions_csv(path: str, account_types: dict[str, str] | None = None) -> pd.DataFrame:
    raw = _read_csv_from_header(path, r"^\"?Account Number")
    col = {c.lower(): c for c in raw.columns}

    def get(name, default=None):
        c = col.get(name.lower())
        return raw[c] if c else pd.Series([default] * len(raw), index=raw.index)

    df = pd.DataFrame({
        "account_id": get("Account Number").str.strip(),
        "account_name": get("Account Name").str.strip(),
        "ticker": get("Symbol").fillna("").str.strip(),
        "description": get("Description"),
        "quantity": _money(get("Quantity")),
        "price": _money(get("Last Price")),
        "market_value": _money(get("Current Value")),
        "cost_basis": _money(get("Cost Basis Total")),
        "sleeve": get("Sleeve name").map(lambda x: x.strip() if isinstance(x, str) and x.strip() else None),
    })
    df = df[df["account_id"].notna() & (df["ticker"] != "")]

    # "Pending Activity" rows are unsettled cash — fold into cash.
    pending = df["ticker"].str.contains("pending", case=False)
    df.loc[pending, ["ticker", "description"]] = ["CASH", "Pending activity"]

    df["asset_kind"] = "equity"
    desc = df["description"].fillna("").str.upper()
    df.loc[desc.str.contains(r"\bETF\b|ISHARES|SPDR|VANGUARD .*ETF|INDEX FD"), "asset_kind"] = "etf"
    df.loc[df["ticker"].str.match(r"^-?[A-Z]+\d{6}[CP]\d+"), "asset_kind"] = "option"
    df.loc[df["ticker"].str.match(r"^[A-Z]{4}X$"), "asset_kind"] = "mutualfund"
    df.loc[df["ticker"].str.match(r"^\d{3}[A-Z0-9]{5}\d$"), "asset_kind"] = "fixed_income"  # CUSIP
    df["account_type"] = None
    df["lots"] = [[] for _ in range(len(df))]

    df = finalize_positions(df, account_types)
    # Merge cash rows per account (core position + pending)
    cash = df[df["asset_kind"] == "cash"]
    if len(cash):
        agg = cash.groupby(["account_id", "account_name", "account_type"], as_index=False)["market_value"].sum()
        agg = agg.assign(ticker="CASH", description="Cash & core position", asset_kind="cash",
                         quantity=agg["market_value"], price=1.0, cost_basis=agg["market_value"],
                         lots=[[] for _ in range(len(agg))], sleeve=None)
        df = pd.concat([df[df["asset_kind"] != "cash"], agg[df.columns]], ignore_index=True)
    logger.info(f"Loaded {len(df)} positions from {os.path.basename(path)}")
    return df


# ---------------------------------------------------------------------------
# Activity history
# ---------------------------------------------------------------------------

_ACTION_RULES = [
    (r"REINVEST", "REI"),
    (r"YOU BOUGHT|^BOUGHT|PURCHASE INTO", "BUY"),
    (r"YOU SOLD|^SOLD|REDEMPTION FROM", "SELL"),
    (r"DIVIDEND RECEIVED|CAP GAIN|LONG-TERM CAP|SHORT-TERM CAP|DIVIDEND", "DIVIDEND"),
    (r"INTEREST", "INTEREST"),
    (r"STOCK SPLIT|FORWARD SPLIT|REVERSE SPLIT", "SPLIT"),
    (r"ELECTRONIC FUNDS TRANSFER RECEIVED|CONTRIBUTION|DIRECT DEPOSIT|CHECK RECEIVED|"
     r"TRANSFERRED FROM|WIRE TRANSFER FROM|DEPOSIT", "CONTRIBUTION"),
    (r"ELECTRONIC FUNDS TRANSFER PAID|TRANSFERRED TO|WIRE TRANSFER TO|DIRECT DEBIT|"
     r"DISTRIBUTION|WITHDRAWAL|CHECK PAID|BILL PAY", "WITHDRAWAL"),
    (r"TRANSFER OF ASSETS.*RECEIVE|ACAT.*RECEIVE|JOURNALED.*IN", "TRANSFER_IN"),
    (r"TRANSFER OF ASSETS.*DELIVER|ACAT.*DELIVER|JOURNALED.*OUT", "TRANSFER_OUT"),
    (r"FEE|COMMISSION|FOREIGN TAX|TAX WITHHELD", "FEE"),
]


def _classify_action(action: str) -> str:
    a = (action or "").upper()
    for pat, typ in _ACTION_RULES:
        if re.search(pat, a):
            return typ
    return "OTHER"


def load_activity_csv(paths: str | list[str]) -> pd.DataFrame:
    """Load one or more Fidelity history exports (they cap each download's date range)."""
    paths = [paths] if isinstance(paths, str) else list(paths)
    frames = []
    for path in paths:
        raw = _read_csv_from_header(path, r"^\"?Run Date")
        col = {c.lower(): c for c in raw.columns}

        def get(*names):
            for n in names:
                c = col.get(n.lower())
                if c:
                    return raw[c]
            return pd.Series([None] * len(raw), index=raw.index)

        acct = get("Account Number")
        if acct.isna().all():
            acct = get("Account")
        action = get("Action").fillna("")
        df = pd.DataFrame({
            "account_id": acct.astype(str).str.strip(),
            "date": pd.to_datetime(get("Run Date"), errors="coerce", format="mixed"),
            "type": action.map(_classify_action),
            "ticker": get("Symbol").fillna("").str.strip().replace("", None),
            "quantity": _money(get("Quantity")),
            "price": _money(get("Price ($)", "Price")),
            "amount": _money(get("Amount ($)", "Amount")),
            "description": action,
        })
        # Mergers / spin-offs / other corporate actions that move shares: Fidelity
        # signs Quantity (+ received, − delivered). Value them as share transfers.
        corp = (df["type"] == "OTHER") & df["ticker"].notna() & df["quantity"].fillna(0).ne(0)
        df.loc[corp & (df["quantity"] > 0), "type"] = "TRANSFER_IN"
        df.loc[corp & (df["quantity"] < 0), "type"] = "TRANSFER_OUT"
        # Share transfers with no symbol are cash transfers
        no_sym = df["ticker"].isna()
        df.loc[no_sym & (df["type"] == "TRANSFER_IN"), "type"] = "CONTRIBUTION"
        df.loc[no_sym & (df["type"] == "TRANSFER_OUT"), "type"] = "WITHDRAWAL"
        df["_file"] = len(frames)
        frames.append(df)

    if not frames:
        return empty_activities()
    allrows = pd.concat(frames, ignore_index=True)
    # Overlapping date ranges repeat rows across files, but identical rows inside one
    # file are genuine (e.g. two same-size fills). Keep each row max(count per file) times.
    key = [c for c in allrows.columns if c != "_file"]
    allrows["_k"] = pd.util.hash_pandas_object(allrows[key].astype(str), index=False)
    allrows["_n"] = allrows.groupby(["_file", "_k"]).cumcount()
    allrows = allrows.drop_duplicates(["_k", "_n"]).drop(columns=["_file", "_k", "_n"])
    acts = finalize_activities(allrows)
    acts = acts[~acts["ticker"].map(is_cash_ticker)].reset_index(drop=True)
    logger.info(f"Loaded {len(acts)} activity rows from {len(paths)} file(s)")
    return acts
