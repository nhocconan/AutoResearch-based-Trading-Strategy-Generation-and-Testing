#!/usr/bin/env python3
"""
update_data.py - Download daily kline data from Binance to fill gaps
=====================================================================
prepare.py only downloads monthly archives. This script fills in the
current month with daily files, then merges into existing Parquets.

Usage:
    python update_data.py                    # Update all symbols/timeframes
    python update_data.py --days 30          # Last 30 days
"""

import argparse
import io
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
import yaml

BASE_URL = "https://data.binance.vision/data/futures/um/daily/klines"
FUNDING_URL = "https://data.binance.vision/data/futures/um/daily/fundingRate"


def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def download_daily_kline(symbol: str, timeframe: str, date_str: str, raw_dir: Path) -> bool:
    """Download a single daily kline file. Returns True if successful."""
    filename = f"{symbol}-{timeframe}-{date_str}.zip"
    url = f"{BASE_URL}/{symbol}/{timeframe}/{filename}"
    dest = raw_dir / "klines" / symbol / timeframe / filename

    if dest.exists():
        return True  # already downloaded

    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            dest.write_bytes(r.content)
            return True
    except Exception:
        pass
    return False


def download_daily_funding(symbol: str, date_str: str, raw_dir: Path) -> bool:
    filename = f"{symbol}-fundingRate-{date_str}.zip"
    url = f"{FUNDING_URL}/{symbol}/{filename}"
    dest = raw_dir / "fundingRate" / symbol / filename

    if dest.exists():
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            dest.write_bytes(r.content)
            return True
    except Exception:
        pass
    return False


def process_kline_zip(zip_path: Path) -> pd.DataFrame:
    """Extract and parse a kline zip file."""
    with zipfile.ZipFile(zip_path) as zf:
        csv_name = zf.namelist()[0]
        with zf.open(csv_name) as f:
            # Check if first line is header
            first_line = f.readline().decode().strip()
            f.seek(0)
            first_val = first_line.split(",")[0]
            try:
                float(first_val)
                has_header = False
            except ValueError:
                has_header = True

            df = pd.read_csv(
                f,
                header=0 if has_header else None,
                names=None if has_header else [
                    "open_time", "open", "high", "low", "close", "volume",
                    "close_time", "quote_volume", "trades",
                    "taker_buy_volume", "taker_buy_quote_volume", "ignore"
                ],
            )

    # Standardize columns
    cols = ["open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_buy_volume", "taker_buy_quote_volume"]
    if len(df.columns) > len(cols):
        df = df.iloc[:, :len(cols)]
    df.columns = cols

    # Parse timestamps
    df["open_time"] = pd.to_datetime(pd.to_numeric(df["open_time"], errors="coerce"), unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(pd.to_numeric(df["close_time"], errors="coerce"), unit="ms", utc=True)
    df = df.dropna(subset=["open_time"])

    for col in ["open", "high", "low", "close", "volume", "quote_volume",
                 "taker_buy_volume", "taker_buy_quote_volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["trades"] = pd.to_numeric(df["trades"], errors="coerce").fillna(0).astype(int)

    return df


def update_parquet(symbol: str, timeframe: str, raw_dir: Path, processed_dir: Path):
    """Merge daily downloads into existing parquet."""
    parquet_path = processed_dir / "klines" / symbol / f"{timeframe}.parquet"
    daily_dir = raw_dir / "klines" / symbol / timeframe

    if not daily_dir.exists():
        return

    # Load existing data
    existing = pd.read_parquet(parquet_path) if parquet_path.exists() else pd.DataFrame()
    max_existing = existing["open_time"].max() if len(existing) > 0 else pd.Timestamp("2000-01-01", tz="UTC")

    # Process new daily files
    new_dfs = []
    for zip_file in sorted(daily_dir.glob("*.zip")):
        try:
            df = process_kline_zip(zip_file)
            # Only keep rows after existing data
            df = df[df["open_time"] > max_existing]
            if len(df) > 0:
                new_dfs.append(df)
        except Exception as e:
            print(f"  Warning: {zip_file.name}: {e}")

    if not new_dfs:
        return 0

    new_data = pd.concat(new_dfs, ignore_index=True)
    combined = pd.concat([existing, new_data], ignore_index=True)
    combined = combined.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(parquet_path, index=False)
    return len(new_data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=30, help="Days to download")
    args = parser.parse_args()

    config = load_config()
    symbols = config["data"]["symbols"]
    timeframes = [tf for tf in config["data"]["timeframes"] if tf != "1m"]  # skip 1m (too large)
    raw_dir = Path(config["data"]["raw_dir"])
    processed_dir = Path(config["data"]["processed_dir"])

    today = datetime.utcnow().date()
    dates = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(args.days)]
    dates.reverse()

    print(f"Downloading daily data: {len(symbols)} symbols × {len(timeframes)} TFs × {len(dates)} days")

    for symbol in symbols:
        print(f"\n[{symbol}]")
        for tf in timeframes:
            downloaded = 0
            for date_str in dates:
                if download_daily_kline(symbol, tf, date_str, raw_dir):
                    downloaded += 1
            new_bars = update_parquet(symbol, tf, raw_dir, processed_dir)
            if new_bars:
                print(f"  {tf}: +{new_bars} new bars")

        # Funding
        for date_str in dates:
            download_daily_funding(symbol, date_str, raw_dir)

    print("\nDone! Run backtests to use updated data.")


if __name__ == "__main__":
    main()
