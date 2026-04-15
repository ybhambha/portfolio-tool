"""
email_alert.py
--------------
Sends rebalance trade instructions via email using SendGrid.

The email includes:
  - Summary of the rebalance signal
  - Full trade list (what to buy, sell, hold)
  - Expected portfolio metrics after rebalancing
  - Next rebalance date

Setup required:
  - SENDGRID_API_KEY in .env
  - ALERT_EMAIL_TO in .env
  - ALERT_EMAIL_FROM in .env
"""

import os
import logging
from datetime import date
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Email template
# ---------------------------------------------------------------------------

def _build_email_html(signal: dict) -> str:
    """Build HTML email body from rebalance signal."""
    today      = signal["signal_date"]
    trade_list = signal["trade_list"]
    metrics    = signal["metrics"]
    summary    = signal["summary"]
    days_next  = signal["days_to_next"]

    # Build trade table rows
    trade_rows = ""
    for _, row in trade_list.iterrows():
        if row["Action"] == "BUY":
            action_color = "#4CAF50"
        elif row["Action"] == "SELL":
            action_color = "#f44336"
        else:
            action_color = "#888"

        weight_sign = "+" if row["Weight Change"] > 0 else ""
        amount_sign = "+" if row["Trade Amount $"] > 0 else ""

        trade_rows += f"""
        <tr>
            <td style="padding:8px;border-bottom:1px solid #eee;
                       font-weight:500">{row['Ticker']}</td>
            <td style="padding:8px;border-bottom:1px solid #eee;
                       text-align:center">{row['Current Weight']}%</td>
            <td style="padding:8px;border-bottom:1px solid #eee;
                       text-align:center">{row['Target Weight']}%</td>
            <td style="padding:8px;border-bottom:1px solid #eee;
                       text-align:center;color:{'#4CAF50' if row['Weight Change']>0 else '#f44336' if row['Weight Change']<0 else '#888'}">
                {weight_sign}{row['Weight Change']}%
            </td>
            <td style="padding:8px;border-bottom:1px solid #eee;
                       text-align:right">${row['Target Value $']:,.2f}</td>
            <td style="padding:8px;border-bottom:1px solid #eee;
                       text-align:right;color:{'#4CAF50' if row['Trade Amount $']>0 else '#f44336' if row['Trade Amount $']<0 else '#888'}">
                {amount_sign}${abs(row['Trade Amount $']):,.2f}
            </td>
            <td style="padding:8px;border-bottom:1px solid #eee;
                       text-align:center">
                <span style="background:{action_color};color:white;
                             padding:2px 10px;border-radius:12px;
                             font-size:12px;font-weight:600">
                    {row['Action']}
                </span>
            </td>
            <td style="padding:8px;border-bottom:1px solid #eee;
                       text-align:right;color:#888;font-size:12px">
                ${row['Est. Cost $']:.2f}
            </td>
        </tr>"""

    total_cost = signal["total_cost"]
    n_trades   = signal["n_trades"]

    html = f"""
<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"></head>
<body style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',
             sans-serif;background:#f5f6fa;margin:0;padding:20px;">

  <div style="max-width:700px;margin:0 auto;">

    <!-- Header -->
    <div style="background:#1a237e;color:white;padding:24px 28px;
                border-radius:10px 10px 0 0;">
      <h1 style="margin:0;font-size:20px;font-weight:600">
        Rebalance Signal
      </h1>
      <p style="margin:6px 0 0;opacity:0.8;font-size:14px">
        MVO Sector ETF Portfolio &nbsp;|&nbsp;
        {today.strftime('%B %d, %Y')}
      </p>
    </div>

    <!-- Summary Cards -->
    <div style="background:white;padding:20px 28px;
                display:flex;gap:16px;flex-wrap:wrap;">
      <div style="flex:1;min-width:140px;background:#e8eaf6;
                  border-radius:8px;padding:14px;text-align:center;">
        <div style="font-size:11px;color:#555;text-transform:uppercase;
                    letter-spacing:0.05em">Trades Required</div>
        <div style="font-size:28px;font-weight:600;color:#1a237e;
                    margin-top:4px">{n_trades}</div>
      </div>
      <div style="flex:1;min-width:140px;background:#e8eaf6;
                  border-radius:8px;padding:14px;text-align:center;">
        <div style="font-size:11px;color:#555;text-transform:uppercase;
                    letter-spacing:0.05em">Est. Cost</div>
        <div style="font-size:28px;font-weight:600;color:#1a237e;
                    margin-top:4px">${total_cost:.2f}</div>
      </div>
      <div style="flex:1;min-width:140px;background:#e8eaf6;
                  border-radius:8px;padding:14px;text-align:center;">
        <div style="font-size:11px;color:#555;text-transform:uppercase;
                    letter-spacing:0.05em">Exp. Return</div>
        <div style="font-size:28px;font-weight:600;color:#1a237e;
                    margin-top:4px">{metrics['expected_return']}%</div>
      </div>
      <div style="flex:1;min-width:140px;background:#e8eaf6;
                  border-radius:8px;padding:14px;text-align:center;">
        <div style="font-size:11px;color:#555;text-transform:uppercase;
                    letter-spacing:0.05em">Exp. Sharpe</div>
        <div style="font-size:28px;font-weight:600;color:#1a237e;
                    margin-top:4px">{metrics['sharpe_ratio']}</div>
      </div>
    </div>

    <!-- Trade List -->
    <div style="background:white;padding:0 28px 24px;border-radius:0;">
      <h2 style="font-size:15px;font-weight:600;color:#1a237e;
                 margin:0 0 12px;padding-top:4px">
        Trade Instructions
      </h2>
      <table style="width:100%;border-collapse:collapse;font-size:13px;">
        <thead>
          <tr style="background:#e8eaf6;">
            <th style="padding:8px;text-align:left;color:#1a237e">ETF</th>
            <th style="padding:8px;text-align:center;color:#1a237e">
                Current %</th>
            <th style="padding:8px;text-align:center;color:#1a237e">
                Target %</th>
            <th style="padding:8px;text-align:center;color:#1a237e">
                Change</th>
            <th style="padding:8px;text-align:right;color:#1a237e">
                Target $</th>
            <th style="padding:8px;text-align:right;color:#1a237e">
                Trade $</th>
            <th style="padding:8px;text-align:center;color:#1a237e">
                Action</th>
            <th style="padding:8px;text-align:right;color:#1a237e">
                Cost $</th>
          </tr>
        </thead>
        <tbody>{trade_rows}</tbody>
        <tfoot>
          <tr style="background:#f5f5f5;">
            <td colspan="6" style="padding:8px;font-weight:600">
                Total estimated cost</td>
            <td colspan="2" style="padding:8px;text-align:right;
                                   font-weight:600">${total_cost:.2f}</td>
          </tr>
        </tfoot>
      </table>
    </div>

    <!-- Next Rebalance -->
    <div style="background:#e8f5e9;padding:14px 28px;
                border-radius:0 0 10px 10px;font-size:13px;color:#2e7d32;">
      Next rebalance in <strong>{days_next} days</strong>.
      This email was generated automatically by your
      MVO Portfolio Construction Tool.
    </div>

  </div>
</body>
</html>"""

    return html


def _build_email_text(signal: dict) -> str:
    """Build plain text fallback for email."""
    trade_list = signal["trade_list"]
    lines = [
        f"REBALANCE SIGNAL — {signal['signal_date'].strftime('%B %d, %Y')}",
        "=" * 50,
        "",
        "TRADE INSTRUCTIONS:",
        "-" * 50,
    ]
    for _, row in trade_list.iterrows():
        lines.append(
            f"{row['Ticker']:<6} {row['Action']:<4} "
            f"${abs(row['Trade Amount $']):>8,.2f}  "
            f"({row['Current Weight']}% → {row['Target Weight']}%)"
        )
    lines += [
        "-" * 50,
        f"Total estimated cost: ${signal['total_cost']:.2f}",
        "",
        f"Expected Return : {signal['metrics']['expected_return']}%",
        f"Expected Sharpe : {signal['metrics']['sharpe_ratio']}",
        "",
        f"Next rebalance in {signal['days_to_next']} days.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Send email via SendGrid
# ---------------------------------------------------------------------------

def send_rebalance_email(signal: dict) -> bool:
    """
    Send rebalance email via SendGrid.

    Returns True if sent successfully, False otherwise.
    Requires SENDGRID_API_KEY, ALERT_EMAIL_TO, ALERT_EMAIL_FROM in .env
    """
    api_key    = os.getenv("SENDGRID_API_KEY", "")
    email_to   = os.getenv("ALERT_EMAIL_TO",   "")
    email_from = os.getenv("ALERT_EMAIL_FROM",  "")

    if not all([api_key, email_to, email_from]):
        logger.warning(
            "Email not sent — missing SENDGRID_API_KEY, "
            "ALERT_EMAIL_TO, or ALERT_EMAIL_FROM in .env"
        )
        print("\nEmail credentials not configured in .env")
        print("To enable email alerts, add to your .env file:")
        print("  SENDGRID_API_KEY=SG.xxxx")
        print("  ALERT_EMAIL_TO=you@example.com")
        print("  ALERT_EMAIL_FROM=alerts@yourdomain.com")
        return False

    try:
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import Mail, Content

        today   = signal["signal_date"]
        subject = (
            f"Portfolio Rebalance Signal — "
            f"{today.strftime('%B %d, %Y')} — "
            f"{signal['n_trades']} trades"
        )

        message = Mail(
            from_email=email_from,
            to_emails=email_to,
            subject=subject,
        )
        message.add_content(
            Content("text/plain", _build_email_text(signal))
        )
        message.add_content(
            Content("text/html", _build_email_html(signal))
        )

        sg       = SendGridAPIClient(api_key)
        response = sg.send(message)

        if response.status_code in [200, 201, 202]:
            logger.info(f"Email sent to {email_to} | "
                        f"Status: {response.status_code}")
            print(f"Email sent successfully to {email_to}")
            return True
        else:
            logger.error(f"Email failed | Status: {response.status_code}")
            return False

    except Exception as e:
        logger.error(f"Email error: {e}")
        print(f"Email error: {e}")
        return False


# ---------------------------------------------------------------------------
# Print trade list to console (used when email is not configured)
# ---------------------------------------------------------------------------

def print_trade_list(signal: dict) -> None:
    """Print trade instructions to console."""
    print("\n" + "=" * 60)
    print(signal["summary"])
    print("=" * 60)

    if not signal.get("is_rebalance"):
        return

    trade_list = signal["trade_list"]
    print("\nTRADE INSTRUCTIONS:")
    print("-" * 60)
    print(f"{'ETF':<6} {'Action':<5} {'Current':>8} {'Target':>8} "
          f"{'Trade $':>10} {'Cost $':>7}")
    print("-" * 60)

    for _, row in trade_list.iterrows():
        sign = "+" if row["Trade Amount $"] > 0 else ""
        print(
            f"{row['Ticker']:<6} "
            f"{row['Action']:<5} "
            f"{row['Current Weight']:>7.2f}% "
            f"{row['Target Weight']:>7.2f}% "
            f"{sign}${abs(row['Trade Amount $']):>8,.2f} "
            f"${row['Est. Cost $']:>5.2f}"
        )

    print("-" * 60)
    print(f"{'TOTAL':<6} {'':<5} {'':<8} {'':<8} "
          f"{'':>10} ${signal['total_cost']:>5.2f}")
    print("=" * 60)
