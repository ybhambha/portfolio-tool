"""
fetch.py
--------
Downloads adjusted closing prices for all ETFs and the benchmark via yfinance.
Caches results to avoid redundant API calls. Supports parquet, SQL Server, and S3 backends.
"""

import os
import logging
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Staleness check
# ---------------------------------------------------------------------------

def is_stale(cached_df: pd.DataFrame, end_date: str, tolerance_days: int = 5) -> bool:
    """Return True if the cached data is missing recent dates."""
    if cached_df is None or cached_df.empty:
        return True
    latest = cached_df.index.max()
    return (pd.Timestamp(end_date) - latest).days > tolerance_days


# ---------------------------------------------------------------------------
# Parquet cache (local)
# ---------------------------------------------------------------------------

def _load_parquet(cache_dir: str, ticker: str) -> pd.DataFrame | None:
    path = os.path.join(cache_dir, f"{ticker}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None


def _save_parquet(cache_dir: str, ticker: str, df: pd.DataFrame) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{ticker}.parquet")
    df.to_parquet(path)
    

# ---------------------------------------------------------------------------
# SQLite backend
# ---------------------------------------------------------------------------

def _get_sqlite_engine(db_path: str):
    """Create a SQLAlchemy engine for SQLite."""
    from sqlalchemy import create_engine
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return create_engine(f"sqlite:///{db_path}")


def _load_sqlite(engine, table: str, ticker: str) -> pd.DataFrame | None:
    """Load a single ticker's price series from SQLite."""
    from sqlalchemy import text, inspect
    inspector = inspect(engine)
    if not inspector.has_table(table):
        return None
    try:
        query = text(f"SELECT date, price FROM {table} WHERE ticker = :ticker ORDER BY date")
        df = pd.read_sql(query, engine, params={"ticker": ticker}, parse_dates=["date"])
        if df.empty:
            return None
        df = df.set_index("date").rename(columns={"price": ticker})
        return df
    except Exception as e:
        logger.warning(f"SQLite load failed for {ticker}: {e}")
        return None


def _save_sqlite(engine, table: str, ticker: str, df: pd.DataFrame) -> None:
    """Save a ticker's price series into SQLite."""
    from sqlalchemy import text

    create_sql = f"""
    CREATE TABLE IF NOT EXISTS {table} (
        ticker  TEXT NOT NULL,
        date    TEXT NOT NULL,
        price   REAL NOT NULL,
        PRIMARY KEY (ticker, date)
    )
    """
    with engine.begin() as conn:
        conn.execute(text(create_sql))
        conn.execute(text(f"DELETE FROM {table} WHERE ticker = :ticker"), {"ticker": ticker})

    records = df.reset_index()
    records.columns = ["date", "price"]
    records["ticker"] = ticker
    records["date"] = records["date"].astype(str)
    records[["ticker", "date", "price"]].to_sql(
        table, engine, if_exists="append", index=False
    )
    logger.info(f"Saved {len(records)} rows for {ticker} to SQLite table '{table}'")


# ---------------------------------------------------------------------------
# SQL Server backend
# ---------------------------------------------------------------------------

def _get_sqlalchemy_engine(db_cfg: dict):
    """Build a SQLAlchemy engine for SQL Server using pyodbc."""
    from sqlalchemy import create_engine
    import urllib

    params = urllib.parse.quote_plus(
        f"DRIVER={{{db_cfg['driver']}}};"
        f"SERVER={db_cfg['host']},{db_cfg['port']};"
        f"DATABASE={db_cfg['database']};"
        f"UID={db_cfg['username']};"
        f"PWD={db_cfg['password']};"
        "TrustServerCertificate=yes;"
    )
    connection_string = f"mssql+pyodbc:///?odbc_connect={params}"
    return create_engine(connection_string, fast_executemany=True)


def _load_sqlserver(engine, table: str, ticker: str) -> pd.DataFrame | None:
    """Load a single ticker's price series from SQL Server."""
    from sqlalchemy import text
    try:
        query = text(f"SELECT date, price FROM {table} WHERE ticker = :ticker ORDER BY date")
        df = pd.read_sql(query, engine, params={"ticker": ticker}, parse_dates=["date"])
        if df.empty:
            return None
        df = df.set_index("date").rename(columns={"price": ticker})
        return df
    except Exception as e:
        logger.warning(f"SQL Server load failed for {ticker}: {e}")
        return None


def _save_sqlserver(engine, table: str, ticker: str, df: pd.DataFrame) -> None:
    """Upsert a ticker's price series into SQL Server."""
    from sqlalchemy import text

    # Ensure table exists
    create_sql = f"""
    IF NOT EXISTS (
        SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = '{table}'
    )
    CREATE TABLE {table} (
        ticker   NVARCHAR(10)   NOT NULL,
        date     DATE           NOT NULL,
        price    FLOAT          NOT NULL,
        PRIMARY KEY (ticker, date)
    )
    """
    with engine.begin() as conn:
        conn.execute(text(create_sql))

    records = df.reset_index().rename(columns={df.columns[0]: "price", "index": "date"})
    records["ticker"] = ticker

    # Delete existing rows for this ticker then re-insert (simple upsert)
    with engine.begin() as conn:
        conn.execute(text(f"DELETE FROM {table} WHERE ticker = :ticker"), {"ticker": ticker})
        records[["ticker", "date", "price"]].to_sql(
            table, conn, if_exists="append", index=False
        )
    logger.info(f"Saved {len(records)} rows for {ticker} to SQL Server table '{table}'")


# ---------------------------------------------------------------------------
# S3 backend
# ---------------------------------------------------------------------------

def _load_s3(bucket: str, prefix: str, ticker: str) -> pd.DataFrame | None:
    """Load a ticker's parquet file from S3."""
    import boto3
    from io import BytesIO

    s3 = boto3.client("s3")
    key = f"{prefix}{ticker}.parquet"
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return pd.read_parquet(BytesIO(obj["Body"].read()))
    except s3.exceptions.NoSuchKey:
        return None
    except Exception as e:
        logger.warning(f"S3 load failed for {ticker}: {e}")
        return None


def _save_s3(bucket: str, prefix: str, ticker: str, df: pd.DataFrame) -> None:
    """Save a ticker's price series as parquet to S3."""
    import boto3
    from io import BytesIO

    s3 = boto3.client("s3")
    key = f"{prefix}{ticker}.parquet"
    buffer = BytesIO()
    df.to_parquet(buffer)
    buffer.seek(0)
    s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
    logger.info(f"Saved {ticker} to s3://{bucket}/{key}")


# ---------------------------------------------------------------------------
# Core fetch function
# ---------------------------------------------------------------------------

def fetch_prices(
    tickers: list[str],
    start: str,
    end: str,
    storage_cfg: dict,
    price_col: str = "Adj Close",
) -> pd.DataFrame:
    """
    Download adjusted closing prices for all tickers.
    Respects the storage backend defined in config.yaml:
      - "parquet"   : local file cache
      - "sqlserver" : SQL Server via pyodbc + SQLAlchemy
      - "s3"        : AWS S3 parquet files

    Parameters
    ----------
    tickers      : list of ticker symbols (ETFs + benchmark)
    start        : start date string "YYYY-MM-DD"
    end          : end date string "YYYY-MM-DD"
    storage_cfg  : the storage block from config.yaml
    price_col    : which yfinance column to use (always "Adj Close")

    Returns
    -------
    pd.DataFrame : shape (trading_days, n_tickers), DatetimeIndex
    """
    backend = storage_cfg.get("backend", "parquet")
    engine = None

    if backend == "sqlserver":
        engine = _get_sqlalchemy_engine(storage_cfg["sqlserver"])
        table = storage_cfg["sqlserver"]["table_prices"]
    elif backend == "sqlite":
        engine = _get_sqlite_engine(storage_cfg["sqlite"]["path"])
        table = storage_cfg["sqlite"]["table_prices"]
    elif backend == "s3":
        s3_cfg = storage_cfg["s3"]

    frames = {}

    for ticker in tickers:
        cached_df = None

        # --- Load from cache ---
        if backend == "parquet":
            cached_df = _load_parquet(storage_cfg.get("cache_dir", "data/cache"), ticker)
        elif backend == "sqlserver":
            cached_df = _load_sqlserver(engine, table, ticker)
        elif backend == "sqlite":
            cached_df = _load_sqlite(engine, table, ticker)
        elif backend == "s3":
            cached_df = _load_s3(s3_cfg["bucket"], s3_cfg["prefix"], ticker)

        # --- Download if missing or stale ---
        if is_stale(cached_df, end):
            logger.info(f"Downloading {ticker} from yfinance ...")
            raw = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
            )
            if raw.empty:
                raise ValueError(f"yfinance returned no data for {ticker}. Check the ticker symbol.")

            # Handle yfinance MultiIndex columns (newer versions return MultiIndex)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.droplevel(1)
            
            series = raw[price_col].copy()
            series.name = ticker
            
            fresh_df = series.to_frame()
            fresh_df.index = pd.to_datetime(fresh_df.index)
            fresh_df.index.name = "date"

            # --- Save to cache ---
            if backend == "parquet":
                _save_parquet(storage_cfg.get("cache_dir", "data/cache"), ticker, fresh_df)
            elif backend == "sqlserver":
                _save_sqlserver(engine, table, ticker, fresh_df)
            elif backend == "sqlite":
                _save_sqlite(engine, table, ticker, fresh_df)
            elif backend == "s3":
                _save_s3(s3_cfg["bucket"], s3_cfg["prefix"], ticker, fresh_df)

            cached_df = fresh_df
        else:
            logger.info(f"Loaded {ticker} from cache ({backend})")

        frames[ticker] = cached_df.iloc[:, 0]

    prices = pd.DataFrame(frames)
    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "date"

    return prices
