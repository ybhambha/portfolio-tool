"""
scheduler.py
------------
Runs a daily check at market close (4:30 PM ET) to determine
whether today is a rebalance day and sends the email alert if so.

Uses APScheduler for scheduling. Runs as a daemon process.

Start with: python run_scheduler.py
"""

import logging
import pytz
from datetime import datetime
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from src.config import load_config
from src.alerts.signals import generate_rebalance_signal
from src.alerts.email_alert import send_rebalance_email, print_trade_list

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def daily_check(
    portfolio_value: float = 10_000.0,
    current_weights: dict | None = None,
    config_path: str = "config.yaml",
) -> None:
    """
    Daily job: check for rebalance signal and send email if triggered.
    Called automatically by the scheduler at 4:30 PM ET every weekday.
    """
    config = load_config(config_path)
    logger.info(f"Running daily check — {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    signal = generate_rebalance_signal(
        config=config,
        portfolio_value=portfolio_value,
        current_weights=current_weights,
    )

    if signal["is_rebalance"]:
        logger.info("Rebalance day detected — generating trade list")
        print_trade_list(signal)
        send_rebalance_email(signal)
    else:
        logger.info(signal["summary"])


def start_scheduler(
    portfolio_value: float = 10_000.0,
    current_weights: dict | None = None,
    config_path: str = "config.yaml",
    hour: int = 16,
    minute: int = 30,
) -> None:
    """
    Start the APScheduler daemon.
    Runs daily_check() every weekday at 4:30 PM Eastern Time.

    Parameters
    ----------
    portfolio_value  : current total portfolio value in dollars
    current_weights  : dict of current weights {ticker: weight}
    config_path      : path to config.yaml
    hour             : hour to run (24h format, ET)
    minute           : minute to run
    """
    eastern = pytz.timezone("US/Eastern")
    scheduler = BlockingScheduler(timezone=eastern)

    scheduler.add_job(
        func=daily_check,
        trigger=CronTrigger(
            day_of_week="mon-fri",  # weekdays only
            hour=hour,
            minute=minute,
            timezone=eastern,
        ),
        kwargs={
            "portfolio_value" : portfolio_value,
            "current_weights" : current_weights,
            "config_path"     : config_path,
        },
        id="daily_rebalance_check",
        name="Daily Rebalance Check",
        replace_existing=True,
    )

    logger.info(
        f"Scheduler started — checking daily at "
        f"{hour:02d}:{minute:02d} ET (weekdays only)"
    )
    logger.info("Press Ctrl+C to stop")

    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user")
        scheduler.shutdown()
