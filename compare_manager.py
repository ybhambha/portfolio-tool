# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 17:20:05 2026

@author: bhamb
"""

import sqlite3
import pandas as pd

# Connect to your SQLite database
conn = sqlite3.connect("data/portfolio.db")

# Fetch SPY prices
spy = pd.read_sql(
    "SELECT date, price FROM etf_prices WHERE ticker='SPY' ORDER BY date",
    conn
)
spy["date"] = pd.to_datetime(spy["date"])
spy = spy.set_index("date")

# Get prices on specific dates
# March 10 2025
mar_10 = spy.loc["2025-03-10", "price"]

# April 17 2026 — use closest available date
apr_17 = spy.loc[spy.index <= "2026-04-17"].iloc[-1]["price"]
apr_17_date = spy.loc[spy.index <= "2026-04-17"].index[-1]

# Your portfolio values
your_start  = 277_455.77
your_end    = 349_281.17
your_return = (your_end - your_start) / your_start * 100

# SPY equivalent
spy_return  = (apr_17 - mar_10) / mar_10 * 100

# What SPY would have grown your money to
spy_end_value = your_start * (1 + spy_return / 100)

print(f"\n{'='*55}")
print(f"PORTFOLIO vs S&P 500 COMPARISON")
print(f"{'='*55}")
print(f"Period: March 10, 2025 → {apr_17_date.date()}")
print(f"\nSPY prices:")
print(f"  March 10, 2025 : ${mar_10:,.2f}")
print(f"  {apr_17_date.date()} : ${apr_17:,.2f}")
print(f"\n{'Metric':<30} {'Your Manager':>15} {'S&P 500':>15}")
print(f"{'-'*60}")
print(f"{'Starting value':<30} ${your_start:>14,.2f} ${your_start:>14,.2f}")
print(f"{'Ending value':<30} ${your_end:>14,.2f} ${spy_end_value:>14,.2f}")
print(f"{'Gain ($)':<30} ${your_end-your_start:>14,.2f} ${spy_end_value-your_start:>14,.2f}")
print(f"{'Total return (%)':<30} {your_return:>14.2f}% {spy_return:>14.2f}%")
print(f"{'='*60}")

conn.close()