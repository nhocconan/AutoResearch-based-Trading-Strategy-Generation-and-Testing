#!/usr/bin/env python3
"""
prepare.py - Data Download and Preprocessing Pipeline
======================================================
IMMUTABLE FILE - Do not modify during research experiments.

Downloads Binance USDT-M futures data (klines, funding rates, metrics)
and processes into efficient Parquet format for backtesting.

Memory-efficient design for 16GB M1 Mac:
- Processes one file at a time
- Uses chunked CSV reading for large files
- Stores as compressed Parquet (typically 3-5x smaller than CSV)

Usage:
    python prepare.py                    # Download + process all
    python prepare.py --download-only    # Just download
    python prepare.py --process-only     # Just process (assumes data downloaded)
    python prepare.py --symbols BTCUSDT  # Single symbol
    python prepare.py --timeframes 1h 4h # Specific timeframes only
"""

import argparse
import hashlib
import io
import os
import sys
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yaml
from tqdm import tqdm

# =============================================================================
# Constants
# =============================================================================

BASE_URL = "https://data.binance.vision/data/futures/um"

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades", "taker_buy_volume",
    "taker_buy_quote_volume", "ignore"
]

KLINE_DTYPES = {
    "open": np.float64,
    "high": np.float64,
    "low": np.float64,
    "close": np.float64,
    "volume": np.float64,
    "quote_volume": np.float64,
    "trades": np.int64,
    "taker_buy_volume": np.float64,
    "taker_buy_quote_volume": np.float64,
}

FUNDING_COLUMNS = ["calc_time", "funding_interval_hours", "last_funding_rate"]


def load_config() -> dict:
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


# =============================================================================
# Download Functions
# =============================================================================

def download_file(url: str, dest_path: Path, verify_checksum: bool = True) -> bool:
    """Download a file from URL to dest_path. Returns True if successful."""
    if dest_path.exists():
        return True

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        resp = requests.get(url, timeout=30, stream=True)
        if resp.status_code == 404:
            return False
        resp.raise_for_status()

        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        # Verify checksum if available
        if verify_checksum:
            checksum_url = f"{url}.CHECKSUM"
            try:
                checksum_resp = requests.get(checksum_url, timeout=10)
                if checksum_resp.status_code == 200:
                    expected_hash = checksum_resp.text.strip().split()[0]
                    actual_hash = hashlib.sha256(dest_path.read_bytes()).hexdigest()
                    if actual_hash != expected_hash:
                        print(f"  CHECKSUM MISMATCH: {dest_path.name}")
                        dest_path.unlink()
                        return False
            except Exception:
                pass  # Checksum verification is best-effort

        return True
    except Exception as e:
        print(f"  Download failed: {url} - {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False


def generate_month_range(start_date: str, end_date: str) -> list[tuple[int, int]]:
    """Generate list of (year, month) tuples between start and end dates."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    months = []
    current = start.replace(day=1)
    while current <= end:
        months.append((current.year, current.month))
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)
    return months


def download_klines(
    symbol: str,
    timeframe: str,
    months: list[tuple[int, int]],
    raw_dir: Path,
) -> list[Path]:
    """Download monthly kline files for a symbol/timeframe."""
    downloaded = []
    desc = f"  {symbol} {timeframe} klines"

    for year, month in tqdm(months, desc=desc, leave=False):
        filename = f"{symbol}-{timeframe}-{year}-{month:02d}.zip"
        url = f"{BASE_URL}/monthly/klines/{symbol}/{timeframe}/{filename}"
        dest = raw_dir / "klines" / symbol / timeframe / filename

        if download_file(url, dest):
            downloaded.append(dest)

    return downloaded


def download_funding_rates(
    symbol: str,
    months: list[tuple[int, int]],
    raw_dir: Path,
) -> list[Path]:
    """Download monthly funding rate files."""
    downloaded = []
    desc = f"  {symbol} funding"

    for year, month in tqdm(months, desc=desc, leave=False):
        filename = f"{symbol}-fundingRate-{year}-{month:02d}.zip"
        url = f"{BASE_URL}/monthly/fundingRate/{symbol}/{filename}"
        dest = raw_dir / "fundingRate" / symbol / filename

        if download_file(url, dest):
            downloaded.append(dest)

    return downloaded


def download_metrics(
    symbol: str,
    start_date: str,
    end_date: str,
    raw_dir: Path,
) -> list[Path]:
    """Download daily metrics files (only available as daily)."""
    downloaded = []
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    delta = end - start

    desc = f"  {symbol} metrics"
    for i in tqdm(range(delta.days + 1), desc=desc, leave=False):
        date = start + timedelta(days=i)
        date_str = date.strftime("%Y-%m-%d")
        filename = f"{symbol}-metrics-{date_str}.zip"
        url = f"{BASE_URL.replace('/monthly', '')}/daily/metrics/{symbol}/{filename}"
        dest = raw_dir / "metrics" / symbol / filename

        if download_file(url, dest, verify_checksum=False):
            downloaded.append(dest)

    return downloaded


def download_all(config: dict, symbols: list[str], timeframes: list[str]):
    """Download all data specified in config."""
    raw_dir = Path(config["data"]["raw_dir"])
    start = config["data"]["train_start"]
    end = config["data"]["test_end"]
    months = generate_month_range(start, end)

    print(f"Downloading data for {len(months)} months ({start} to {end})")
    print(f"Symbols: {symbols}")
    print(f"Timeframes: {timeframes}")
    print()

    for symbol in symbols:
        print(f"[{symbol}]")

        # Klines
        for tf in timeframes:
            download_klines(symbol, tf, months, raw_dir)

        # Funding rates
        download_funding_rates(symbol, months, raw_dir)

        # Metrics (daily - takes longer, download separately if needed)
        # Uncomment to download metrics:
        # download_metrics(symbol, start, end, raw_dir)

        print()


# =============================================================================
# Processing Functions
# =============================================================================

def extract_csv_from_zip(zip_path: Path) -> Optional[pd.DataFrame]:
    """Extract CSV from a zip file. Returns DataFrame or None."""
    try:
        with zipfile.ZipFile(zip_path) as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            if not csv_names:
                return None
            with zf.open(csv_names[0]) as f:
                return pd.read_csv(f)
    except (zipfile.BadZipFile, Exception) as e:
        print(f"  Error extracting {zip_path.name}: {e}")
        return None


def process_klines(
    symbol: str,
    timeframe: str,
    raw_dir: Path,
    processed_dir: Path,
) -> Optional[Path]:
    """Process raw kline zips into a single Parquet file."""
    zip_dir = raw_dir / "klines" / symbol / timeframe
    if not zip_dir.exists():
        return None

    zip_files = sorted(zip_dir.glob("*.zip"))
    if not zip_files:
        return None

    dfs = []
    for zf in zip_files:
        try:
            with zipfile.ZipFile(zf) as z:
                csv_names = [n for n in z.namelist() if n.endswith(".csv")]
                if not csv_names:
                    continue
                with z.open(csv_names[0]) as f:
                    # Read first line to detect if there's a header
                    first_line = f.readline().decode("utf-8").strip()
                    f.seek(0)

                    # If first field is not numeric, file has a header row
                    first_field = first_line.split(",")[0]
                    has_header = not first_field.replace(".", "").replace("-", "").isdigit()

                    if has_header:
                        df = pd.read_csv(f)
                        # Rename columns to our standard names
                        df.columns = KLINE_COLUMNS[:len(df.columns)]
                    else:
                        df = pd.read_csv(f, header=None, names=KLINE_COLUMNS)
                    dfs.append(df)
        except Exception as e:
            print(f"  Error processing {zf.name}: {e}")
            continue

    if not dfs:
        return None

    # Concatenate and clean
    df = pd.concat(dfs, ignore_index=True)

    # Ensure open_time is numeric before converting to datetime
    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")
    df["close_time"] = pd.to_numeric(df["close_time"], errors="coerce")
    df = df.dropna(subset=["open_time"])

    # Convert timestamps
    df["open_time"] = pd.to_datetime(df["open_time"].astype(np.int64), unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"].astype(np.int64), unit="ms", utc=True)

    # Apply dtypes
    for col, dtype in KLINE_DTYPES.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(dtype)

    # Drop duplicates and sort
    df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)

    # Drop the 'ignore' column
    df = df.drop(columns=["ignore"], errors="ignore")

    # Save as Parquet
    out_dir = processed_dir / "klines" / symbol
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{timeframe}.parquet"
    df.to_parquet(out_path, engine="pyarrow", compression="snappy")

    print(f"  {symbol}/{timeframe}: {len(df):,} bars -> {out_path.name} ({out_path.stat().st_size / 1024:.0f} KB)")
    return out_path


def process_funding_rates(
    symbol: str,
    raw_dir: Path,
    processed_dir: Path,
) -> Optional[Path]:
    """Process funding rate zips into a single Parquet file."""
    zip_dir = raw_dir / "fundingRate" / symbol
    if not zip_dir.exists():
        return None

    zip_files = sorted(zip_dir.glob("*.zip"))
    if not zip_files:
        return None

    dfs = []
    for zf in zip_files:
        try:
            with zipfile.ZipFile(zf) as z:
                csv_names = [n for n in z.namelist() if n.endswith(".csv")]
                if not csv_names:
                    continue
                with z.open(csv_names[0]) as f:
                    df = pd.read_csv(f)
                    # Normalize column names
                    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
                    dfs.append(df)
        except Exception as e:
            print(f"  Error processing {zf.name}: {e}")
            continue

    if not dfs:
        return None

    df = pd.concat(dfs, ignore_index=True)

    # Normalize column names to expected format
    col_map = {}
    for col in df.columns:
        if "calc" in col and "time" in col:
            col_map[col] = "calc_time"
        elif "interval" in col:
            col_map[col] = "funding_interval_hours"
        elif "funding" in col and "rate" in col:
            col_map[col] = "last_funding_rate"
    if col_map:
        df = df.rename(columns=col_map)

    # Convert timestamp
    if "calc_time" in df.columns:
        df["calc_time"] = pd.to_datetime(df["calc_time"], unit="ms", utc=True)
        df["last_funding_rate"] = df["last_funding_rate"].astype(np.float64)
        df = df.drop_duplicates(subset=["calc_time"]).sort_values("calc_time").reset_index(drop=True)

    # Save
    out_dir = processed_dir / "funding" / symbol
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "funding_rate.parquet"
    df.to_parquet(out_path, engine="pyarrow", compression="snappy")

    print(f"  {symbol} funding: {len(df):,} records -> {out_path.name}")
    return out_path


def process_all(config: dict, symbols: list[str], timeframes: list[str]):
    """Process all downloaded data into Parquet files."""
    raw_dir = Path(config["data"]["raw_dir"])
    processed_dir = Path(config["data"]["processed_dir"])

    print("Processing data into Parquet format...")
    print()

    for symbol in symbols:
        print(f"[{symbol}]")

        # Process klines
        for tf in timeframes:
            process_klines(symbol, tf, raw_dir, processed_dir)

        # Process funding rates
        process_funding_rates(symbol, raw_dir, processed_dir)

        print()


# =============================================================================
# Data Loading (Used by backtest.py)
# =============================================================================

def load_klines(
    symbol: str,
    timeframe: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    config: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Load processed kline data for a symbol/timeframe.

    Returns DataFrame with columns:
        open_time, open, high, low, close, volume, quote_volume,
        trades, taker_buy_volume, taker_buy_quote_volume, close_time
    """
    if config is None:
        config = load_config()

    processed_dir = Path(config["data"]["processed_dir"])
    path = processed_dir / "klines" / symbol / f"{timeframe}.parquet"

    if not path.exists():
        raise FileNotFoundError(
            f"Data not found: {path}. Run 'python prepare.py' first."
        )

    df = pd.read_parquet(path)

    # Filter by date range
    if start_date:
        start_ts = pd.Timestamp(start_date, tz="UTC")
        df = df[df["open_time"] >= start_ts]
    if end_date:
        end_ts = pd.Timestamp(end_date, tz="UTC")
        df = df[df["open_time"] <= end_ts]

    return df.reset_index(drop=True)


def load_funding_rate(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    config: Optional[dict] = None,
) -> pd.DataFrame:
    """Load processed funding rate data for a symbol."""
    if config is None:
        config = load_config()

    processed_dir = Path(config["data"]["processed_dir"])
    path = processed_dir / "funding" / symbol / "funding_rate.parquet"

    if not path.exists():
        raise FileNotFoundError(
            f"Funding data not found: {path}. Run 'python prepare.py' first."
        )

    df = pd.read_parquet(path)

    if start_date:
        start_ts = pd.Timestamp(start_date, tz="UTC")
        df = df[df["calc_time"] >= start_ts]
    if end_date:
        end_ts = pd.Timestamp(end_date, tz="UTC")
        df = df[df["calc_time"] <= end_ts]

    return df.reset_index(drop=True)


def get_train_test_split(config: Optional[dict] = None) -> dict:
    """Return train/test date boundaries from config."""
    if config is None:
        config = load_config()
    return {
        "train_start": config["data"]["train_start"],
        "train_end": config["data"]["train_end"],
        "test_start": config["data"]["test_start"],
        "test_end": config["data"]["test_end"],
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Download and process Binance futures data")
    parser.add_argument("--download-only", action="store_true", help="Only download, skip processing")
    parser.add_argument("--process-only", action="store_true", help="Only process, skip download")
    parser.add_argument("--symbols", nargs="+", default=None, help="Symbols to process")
    parser.add_argument("--timeframes", nargs="+", default=None, help="Timeframes to process")
    parser.add_argument("--with-metrics", action="store_true", help="Also download daily metrics (slow)")
    args = parser.parse_args()

    config = load_config()
    symbols = args.symbols or config["data"]["symbols"]
    timeframes = args.timeframes or config["data"]["timeframes"]

    print("=" * 60)
    print("LLM Trading Research - Data Preparation")
    print("=" * 60)
    print()

    if not args.process_only:
        download_all(config, symbols, timeframes)

        # Download metrics if requested
        if args.with_metrics:
            raw_dir = Path(config["data"]["raw_dir"])
            for symbol in symbols:
                print(f"Downloading metrics for {symbol}...")
                download_metrics(
                    symbol,
                    config["data"]["train_start"],
                    config["data"]["test_end"],
                    raw_dir,
                )

    if not args.download_only:
        process_all(config, symbols, timeframes)

    print("Done!")


if __name__ == "__main__":
    main()
