"""
run_scheduler.py
----------------
Starts the APScheduler daemon that runs a daily check
at 4:30 PM Eastern Time every weekday.

On rebalance days it automatically:
  1. Fetches latest prices
  2. Computes new optimal weights
  3. Generates trade list
  4. Sends email alert (if configured in .env)

Run with: python run_scheduler.py
Stop with: Ctrl+C
"""

from src.alerts.scheduler import start_scheduler

# -----------------------------------------------------------------------
# Configure these before running
# -----------------------------------------------------------------------

PORTFOLIO_VALUE = 10_000.0   # your total portfolio value in dollars
CURRENT_WEIGHTS = None        # None = assume equal weights

# -----------------------------------------------------------------------

if __name__ == "__main__":
    print("Starting portfolio rebalance scheduler...")
    print("Checks daily at 4:30 PM ET on weekdays")
    print("Press Ctrl+C to stop\n")

    start_scheduler(
        portfolio_value=PORTFOLIO_VALUE,
        current_weights=CURRENT_WEIGHTS,
        config_path="config.yaml",
        hour=16,
        minute=30,
    )
