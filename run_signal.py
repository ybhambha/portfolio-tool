"""
run_signal.py
-------------
One-off entry point to check today's rebalance signal
and print the trade list to the console.

Use this to manually check if today is a rebalance day
and what trades need to be executed.

Run with: python run_signal.py

Optional arguments (edit below):
  PORTFOLIO_VALUE  : your current portfolio value in dollars
  CURRENT_WEIGHTS  : your current ETF weights (if None, assumes equal)
"""

from src.config import load_config
from src.alerts.signals import generate_rebalance_signal
from src.alerts.email_alert import send_rebalance_email, print_trade_list

# -----------------------------------------------------------------------
# Configure these values before running
# -----------------------------------------------------------------------

PORTFOLIO_VALUE = 10_000.0   # your total portfolio value in dollars

# Your current weights — update these to match your actual holdings.
# If None, the signal generator assumes equal weights across all ETFs.
# Example:
# CURRENT_WEIGHTS = {
#     "XLK" : 0.30,
#     "XLF" : 0.15,
#     "XLE" : 0.10,
#     "XLV" : 0.10,
#     "XLC" : 0.10,
#     "XLI" : 0.05,
#     "XLY" : 0.05,
#     "XLP" : 0.05,
#     "XLRE": 0.05,
#     "XLB" : 0.03,
#     "XLU" : 0.02,
# }
CURRENT_WEIGHTS = None   # set to None to assume equal weights

SEND_EMAIL = True   # set to True to also send email alert

# -----------------------------------------------------------------------


def main():
    config = load_config("config.yaml")

    print(f"Checking rebalance signal for today...")
    print(f"Portfolio value : ${PORTFOLIO_VALUE:,.2f}")
    print(f"Rebalance freq  : {config.optimizer.rebalance_frequency}")

    signal = generate_rebalance_signal(
        config=config,
        portfolio_value=PORTFOLIO_VALUE,
        current_weights=CURRENT_WEIGHTS,
    )

    # Always print to console
    print_trade_list(signal)

    # Optionally send email
    if SEND_EMAIL and signal["is_rebalance"]:
        print("\nSending email alert...")
        send_rebalance_email(signal)


if __name__ == "__main__":
    main()
