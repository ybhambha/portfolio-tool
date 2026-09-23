"""
config.py
---------
Loads config.yaml and injects secrets from environment variables (.env file).
Uses Pydantic for validation so bad config values fail fast with clear messages.
"""

import os
import yaml
from datetime import date
from dotenv import load_dotenv
from pydantic import BaseModel, field_validator

load_dotenv()  # loads .env into os.environ


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class UniverseConfig(BaseModel):
    etfs: list[str]
    benchmark: str

class SQLiteConfig(BaseModel):
    path: str = "data/portfolio.db"
    table_prices: str = "etf_prices"
    table_returns: str = "etf_returns"

class SqlServerConfig(BaseModel):
    host: str = "localhost"
    port: int = 1433
    database: str = "portfolio_db"
    username: str = ""
    password: str = ""
    driver: str = "ODBC Driver 18 for SQL Server"
    table_prices: str = "etf_prices"
    table_returns: str = "etf_returns"

    def model_post_init(self, __context):
        # Override credentials from environment
        self.username = os.getenv("DB_USERNAME", self.username)
        self.password = os.getenv("DB_PASSWORD", self.password)


class S3Config(BaseModel):
    bucket: str = ""
    prefix: str = "cache/"
    region: str = "us-east-1"


class StorageConfig(BaseModel):
    backend: str = "sqlite"
    cache_dir: str = "data/cache"
    sqlserver: SqlServerConfig = SqlServerConfig()
    sqlite: SQLiteConfig = SQLiteConfig()
    s3: S3Config = S3Config()

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, v):
        allowed = {"parquet", "sqlserver", "sqlite", "s3"}
        if v not in allowed:
            raise ValueError(f"storage.backend must be one of {allowed}, got '{v}'")
        return v


class DataConfig(BaseModel):
    start_date: str
    end_date: str
    price_col: str = "Adj Close"
    missing_threshold: float = 0.02

    def model_post_init(self, __context):
        if self.end_date.lower() == "today":
            self.end_date = date.today().strftime("%Y-%m-%d")


class OptimizerConfig(BaseModel):
    lookback_days: int = 252
    rebalance_frequency: str = "monthly"
    weight_min: float = 0.0
    weight_max: float = 0.40
    transaction_cost_bps: float = 5.0


class AlertsConfig(BaseModel):
    rebalance_trigger: str = "calendar"
    drift_threshold: float = 0.05
    email_to: str = ""
    email_from: str = ""

    def model_post_init(self, __context):
        self.email_to = os.getenv("ALERT_EMAIL_TO", self.email_to)
        self.email_from = os.getenv("ALERT_EMAIL_FROM", self.email_from)


class PerformanceConfig(BaseModel):
    risk_free_rate: float = 0.05

    def model_post_init(self, __context):
        env_val = os.getenv("RISK_FREE_RATE")
        if env_val:
            self.risk_free_rate = float(env_val)


class FidelityRebalanceConfig(BaseModel):
    target_mode: str = "mvo"                  # "mvo" | "static"
    model_universe: list[str] = []            # empty → optimize over current holdings
    static_targets: dict[str, float] = {}
    unmanaged: str = "hold"                   # "hold" | "sell"
    accounts: list[str] = []                  # empty → all accounts
    managed_accounts: list[str] = []          # advisor-managed accounts to leave untouched
    exclude_managed_sleeves: bool = True      # skip Fidelity Strategic Advisers SMA sleeves
    max_mvo_assets: int = 40
    cash_target_pct: float = 0.02
    drift_band: float = 0.02
    min_trade_usd: float = 100.0
    max_turnover: float = 0.30
    fractional_shares: bool = False
    avoid_short_term_gains: bool = True
    tlh_loss_pct: float = 0.05
    tlh_min_usd: float = 250.0
    risk_aversion: float = 1.0

    @field_validator("target_mode")
    @classmethod
    def validate_mode(cls, v):
        if v not in {"mvo", "static"}:
            raise ValueError("fidelity.rebalance.target_mode must be 'mvo' or 'static'")
        return v

    @field_validator("unmanaged")
    @classmethod
    def validate_unmanaged(cls, v):
        if v not in {"hold", "sell"}:
            raise ValueError("fidelity.rebalance.unmanaged must be 'hold' or 'sell'")
        return v


class FidelityConfig(BaseModel):
    source: str = "auto"                      # "snaptrade" | "csv" | "auto"
    csv_dir: str = "data/fidelity"            # where Fidelity downloads are saved
    positions_glob: str = "Portfolio_Positions*.csv"
    history_glob: str = "Accounts_History*.csv"
    snaptrade_broker: str = "FIDELITY"
    history_start: str | None = None          # None → earliest available
    analytics_lookback_days: int = 756
    account_types: dict[str, str] = {}        # override: {"Z12345678": "tax_deferred"}
    output_dir: str = "data/reports"
    rebalance: FidelityRebalanceConfig = FidelityRebalanceConfig()

    @field_validator("source")
    @classmethod
    def validate_source(cls, v):
        if v not in {"snaptrade", "csv", "auto"}:
            raise ValueError("fidelity.source must be 'snaptrade', 'csv' or 'auto'")
        return v


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

class AppConfig(BaseModel):
    universe: UniverseConfig
    data: DataConfig
    storage: StorageConfig
    optimizer: OptimizerConfig
    alerts: AlertsConfig
    performance: PerformanceConfig
    fidelity: FidelityConfig = FidelityConfig()

    @property
    def all_tickers(self) -> list[str]:
        return self.universe.etfs + [self.universe.benchmark]


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str = "config.yaml") -> AppConfig:
    """Load and validate the full application config."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    # Private overrides (account numbers etc.) live in config.local.yaml next to
    # config.yaml. It is git-ignored so personal details never reach GitHub.
    local_path = os.path.join(os.path.dirname(os.path.abspath(path)), "config.local.yaml")
    if os.path.exists(local_path):
        with open(local_path) as f:
            raw = _deep_merge(raw, yaml.safe_load(f) or {})

    config = AppConfig(
        universe=raw["universe"],
        data=raw["data"],
        storage=raw.get("storage", {}),
        optimizer=raw.get("optimizer", {}),
        alerts=raw.get("alerts", {}),
        performance=raw.get("performance", {}),
        fidelity=raw.get("fidelity") or {},
    )
    return config
