# MVO Sector ETF Portfolio Construction Tool

A mean-variance optimized portfolio construction tool that builds
optimized portfolios comprised of SPY sector ETFs, backtests them
against SPY, and provides automated monthly rebalance alerts.

---

## Features

- **Mean-Variance Optimization** using Ledoit-Wolf covariance shrinkage
- **Walk-forward backtesting** with no look-ahead bias
- **Performance analytics**: annualized return, Sharpe ratio, drawdown, Calmar ratio
- **HTML performance report** with embedded charts
- **Monthly rebalance alerts** via email (SendGrid)
- **Parameter sweep** to find optimal weight constraints
- **SQLite or parquet** data storage
- **Docker** containerization for easy deployment

---

## Project Structure

```
portfolio_tool/
├── config.yaml              ← strategy parameters
├── .env                     ← secrets (never commit)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── src/
│   ├── main.py              ← CLI entry point
│   ├── config.py            ← config loader
│   ├── data/                ← Phase 1: data pipeline
│   ├── optimizer/           ← Phase 2: MVO engine
│   ├── backtest/            ← Phase 3: backtesting
│   ├── reporting/           ← Phase 4: HTML report
│   ├── alerts/              ← Phase 5: trade signals & email
│   └── fidelity/            ← Fidelity positions, performance, rebalance
├── tests/                   ← pytest unit tests
├── data/
│   ├── cache/               ← parquet price cache
│   ├── charts/              ← saved chart images
│   └── reports/             ← generated HTML reports
└── run_*.py                 ← individual phase entry points
```

---

## Quick Start

### 1. Setup

```bash
# Clone or unzip the project
cd portfolio_tool

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate     # Windows
source .venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Copy and configure secrets
copy .env.example .env     # Windows
cp .env.example .env       # Mac/Linux
# Edit .env with your credentials
```

### 2. Configure strategy

Edit `config.yaml`:
```yaml
optimizer:
  weight_max: 0.30           # max 30% per ETF
  rebalance_frequency: monthly
  transaction_cost_bps: 5
```

### 3. Run each phase

```bash
# Phase 1: Fetch data
python run_data.py

# Phase 2: Run optimizer (sanity check)
python run_optimizer.py

# Phase 3: Full backtest
python run_backtest.py

# Phase 4: Generate HTML report
python run_report.py

# Phase 5: Check today's signal
python run_signal.py

# Phase 5: Start daily scheduler
python run_scheduler.py

# Parameter sweep
python run_parameter_sweep.py
```

### 4. Run via CLI

```bash
python -m src.main run-data
python -m src.main run-backtest
python -m src.main run-report
python -m src.main run-signal --value 50000
python -m src.main run-scheduler
```

---

## Fidelity Personal Portfolio

Pulls your real Fidelity positions and history, measures performance, and
produces a tax-aware rebalance trade list from the same MVO engine.
**Read-only — it never places orders**; you review and trade in Fidelity.

### Data source (pick one)

**A. SnapTrade API (automatic)** — Fidelity has no public retail API; SnapTrade
connects through Fidelity's own login.

```bash
# 1. Free personal key at https://dashboard.snaptrade.com → add to .env:
#    SNAPTRADE_CLIENT_ID=...  SNAPTRADE_CONSUMER_KEY=...
python -m src.main fidelity-connect      # open link, log in to Fidelity (read-only)
python -m src.main fidelity-positions    # sanity check
```

**B. CSV export (no third party)** — save into `data/fidelity/`:
- Positions → Download → `Portfolio_Positions_*.csv`
- Activity & Orders → History → Download → `Accounts_History*.csv` (download
  several date ranges if you like; duplicates are removed)

`fidelity.source: auto` uses SnapTrade when keys are in `.env`, otherwise CSV.

### Run

```bash
python -m src.main fidelity-report                       # report + trade list
python -m src.main fidelity-report --source csv --no-rebalance
```

Outputs in `data/reports/`: `fidelity_report_YYYYMMDD.html`,
`fidelity_trades_YYYYMMDD.csv`, `fidelity_positions_YYYYMMDD.csv`.

### What's in the report

| Section | Method |
|---|---|
| Holdings | All accounts, cost basis, unrealized gain, account tax type |
| Realized performance | Daily holdings rebuilt **backwards from today's shares** using the history (so partial history still reconciles). Time-weighted return, money-weighted (XIRR), benchmark over the same dates, and a public-market-equivalent: the same cash flows put into SPY |
| Risk profile | Today's weights on 3 years of prices: CAGR, vol, Sharpe, drawdown, beta, tracking error, risk contribution per holding |
| Rebalance | MVO (Ledoit-Wolf) or static targets → drift band, min trade, turnover cap, cash buffer. Sells come from IRAs/Roth first, then taxable lots with losses; buys go where the cash is. Flags estimated gains, short-term lots, wash sales, and tax-loss-harvest candidates |

Configure under `fidelity:` in `config.yaml` (model universe, static targets,
bands, accounts to include, account-type overrides).

**Limits:** options and individual bonds are reported but not modelled;
short/long-term tax splits need lot dates (SnapTrade paid tier). Fidelity CSVs
only give average cost; tax figures are estimates, not tax advice.

---

## Docker

### Build and run locally

```bash
# Build image
docker build -t portfolio-tool .

# Check today's signal
docker compose run --rm signal

# Start daily scheduler daemon
docker compose up scheduler

# Run backtest
docker compose run --rm backtest

# Generate report
docker compose run --rm report
```

### Deploy to AWS

1. Push image to ECR:
```bash
aws ecr create-repository --repository-name portfolio-tool
aws ecr get-login-password | docker login --username AWS \
    --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com
docker tag portfolio-tool:latest \
    <account>.dkr.ecr.us-east-1.amazonaws.com/portfolio-tool:latest
docker push \
    <account>.dkr.ecr.us-east-1.amazonaws.com/portfolio-tool:latest
```

2. Create ECS Fargate task pointing at your ECR image
3. Store secrets in AWS Secrets Manager
4. Use EventBridge Scheduler to trigger daily at market close

---

## Configuration

### config.yaml

| Parameter | Default | Description |
|---|---|---|
| `etfs` | 11 sector ETFs | Universe of ETFs |
| `benchmark` | SPY | Benchmark for comparison |
| `start_date` | 2015-10-08 | Data start date |
| `lookback_days` | 252 | Training window (1 year) |
| `rebalance_frequency` | monthly | monthly or quarterly |
| `weight_min` | 0.0 | Min weight per ETF (long-only) |
| `weight_max` | 0.30 | Max weight per ETF |
| `transaction_cost_bps` | 5 | Cost per trade in basis points |
| `risk_free_rate` | 0.05 | Annualized risk-free rate |

### config.local.yaml (private, git-ignored)

Personal settings such as Fidelity account numbers go in `config.local.yaml`
(copy `config.local.example.yaml`). It is merged over `config.yaml` at load
time and never committed.

### .env secrets

```bash
SENDGRID_API_KEY=SG.xxxx      # for email alerts
ALERT_EMAIL_TO=you@example.com
ALERT_EMAIL_FROM=alerts@yourdomain.com
RISK_FREE_RATE=0.05
DB_USERNAME=sa                 # if using SQL Server
DB_PASSWORD=yourpassword
```

---

## Running Tests

```bash
pytest tests/ -v
```

Run the full suite with `pytest tests/ -v`; the Fidelity tests need no network.

---

## Backtest Results (2019-2024)

| Metric | Portfolio | SPY | Difference |
|---|---|---|---|
| Annualized Return | 17.27% | 15.28% | +1.99% |
| Annualized Volatility | 19.91% | 20.35% | -0.44% |
| Sharpe Ratio | 0.664 | 0.560 | +0.104 |
| Max Drawdown | -30.50% | -33.72% | +3.22% |
| Avg Drawdown | -4.02% | -6.86% | +2.84% |
| Calmar Ratio | 0.566 | 0.453 | +0.113 |
| Monthly Win Rate | 52.2% | — | — |

*Strategy: 30% max weight, monthly rebalancing, 5 bps transaction cost*
