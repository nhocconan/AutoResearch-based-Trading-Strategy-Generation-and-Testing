#!/usr/bin/env python3
"""
mtf_data.py - Multi-Timeframe Data Loader
==========================================
Loads ACTUAL Binance higher-timeframe data from downloaded Parquet files.
DO NOT resample lower timeframes — use this module instead.

Usage in strategy.py:
    from mtf_data import get_htf_data

    def generate_signals(prices):
        # prices is your primary timeframe (e.g., 1h)
        # Load actual 4h Binance data
        df_4h = get_htf_data(prices, '4h')

        # df_4h has: open_time, open, high, low, close, volume
        # Aligned: for each row in prices, get the LAST COMPLETED 4h bar
        trend_4h = compute_your_indicator(df_4h)

        # Map back to primary timeframe using merge_asof
        trend_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
"""

from pathlib import Path
import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).parent / "data" / "processed" / "klines"


def detect_symbol(prices: pd.DataFrame) -> str:
    """Detect which symbol this prices DataFrame belongs to."""
    if "open_time" not in prices.columns:
        return ""

    first_close = float(prices["close"].iloc[100])  # Use bar 100 to avoid edge cases
    first_time = prices["open_time"].iloc[100]

    for symbol_dir in sorted(DATA_DIR.iterdir()):
        if not symbol_dir.is_dir():
            continue
        # Check 1h data (most common primary TF)
        for tf in ["1h", "15m", "30m", "5m", "4h", "6h", "12h", "1d"]:
            tf_path = symbol_dir / f"{tf}.parquet"
            if tf_path.exists():
                df = pd.read_parquet(tf_path)
                # Find matching time
                match = df[df["open_time"] == first_time]
                if len(match) > 0:
                    ref_close = float(match["close"].iloc[0])
                    if abs(ref_close - first_close) / first_close < 0.001:
                        return symbol_dir.name
    return ""


def get_htf_data(prices: pd.DataFrame, htf: str = "4h") -> pd.DataFrame:
    """
    Load actual Binance higher-timeframe data, filtered to the same time range.

    Args:
        prices: Primary timeframe DataFrame (must have 'open_time' column)
        htf: Target higher timeframe ('4h', '1d', '1h', '15m', etc.)

    Returns:
        DataFrame with actual Binance OHLCV data for the higher timeframe,
        filtered to the same date range as prices.
    """
    symbol = detect_symbol(prices)
    if not symbol:
        raise ValueError("Could not detect symbol from prices data")

    htf_path = DATA_DIR / symbol / f"{htf}.parquet"
    if not htf_path.exists():
        raise FileNotFoundError(f"HTF data not found: {htf_path}")

    df = pd.read_parquet(htf_path)

    # Filter to same time range (with some buffer for lookback)
    min_time = prices["open_time"].min()
    max_time = prices["open_time"].max()
    df = df[(df["open_time"] >= min_time) & (df["open_time"] <= max_time)]

    return df.reset_index(drop=True)


def align_htf_to_ltf(
    prices: pd.DataFrame,
    htf_df: pd.DataFrame,
    htf_values: np.ndarray,
    use_completed_only: bool = True,
) -> np.ndarray:
    """
    Map higher-timeframe values back to lower-timeframe bars.

    CRITICAL: use_completed_only=True (default) ensures we only use the LAST
    COMPLETED HTF bar, not the current one still forming. This prevents look-ahead.

    Args:
        prices: Primary (lower) timeframe DataFrame
        htf_df: Higher timeframe DataFrame (from get_htf_data)
        htf_values: numpy array of indicator values, same length as htf_df
        use_completed_only: If True, shift HTF values by 1 to avoid using unclosed bar

    Returns:
        numpy array of same length as prices, with HTF values forward-filled
    """
    # Create a Series indexed by HTF open_time
    htf_series = pd.Series(htf_values, index=htf_df["open_time"])

    if use_completed_only:
        # Shift by 1 HTF bar: at any LTF bar, only use the PREVIOUS completed HTF bar
        htf_series = htf_series.shift(1)

    # Reindex to LTF timestamps with forward-fill
    ltf_times = prices["open_time"]
    aligned = htf_series.reindex(ltf_times, method="ffill")

    return aligned.values


def get_htf_indicator(
    prices: pd.DataFrame,
    htf: str,
    indicator_fn,
    use_completed_only: bool = True,
) -> np.ndarray:
    """
    Convenience: load HTF data, compute indicator, align back to LTF.

    Args:
        prices: Primary timeframe DataFrame
        htf: Higher timeframe string ('4h', '1d', etc.)
        indicator_fn: Function that takes HTF DataFrame and returns numpy array
        use_completed_only: Only use completed HTF bars (prevents look-ahead)

    Returns:
        numpy array aligned to prices timeframe

    Example:
        def compute_hma(df):
            # compute HMA on df['close']
            return hma_values

        hma_4h = get_htf_indicator(prices, '4h', compute_hma)
    """
    htf_df = get_htf_data(prices, htf)
    htf_values = indicator_fn(htf_df)
    return align_htf_to_ltf(prices, htf_df, htf_values, use_completed_only)
