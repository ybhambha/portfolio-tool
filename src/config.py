"""
config.py
---------
Loads config.yaml and injects secrets from environment variables (.env file).
Uses Pydantic for validation so bad config values fail fast with clear messages.
"""

import os
import yaml
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

    @property
    def all_tickers(self) -> list[str]:
        return self.universe.etfs + [self.universe.benchmark]


def load_config(path: str = "config.yaml") -> AppConfig:
    """Load and validate the full application config."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    config = AppConfig(
        universe=raw["universe"],
        data=raw["data"],
        storage=raw.get("storage", {}),
        optimizer=raw.get("optimizer", {}),
        alerts=raw.get("alerts", {}),
        performance=raw.get("performance", {}),
    )
    return config
