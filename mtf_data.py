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
    if len(prices) == 0:
        return ""

    sample_idx = min(100, len(prices) - 1)  # Use bar 100 when possible, else last available
    first_close = float(prices["close"].iloc[sample_idx])
    first_time = prices["open_time"].iloc[sample_idx]

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

    # Load ALL data up to prices max time (includes warmup before period start)
    # This ensures indicators on HTF have full warmup regardless of period
    max_time = prices["open_time"].max()
    df = df[df["open_time"] <= max_time]

    return df.reset_index(drop=True)


def align_htf_to_ltf(
    prices: pd.DataFrame,
    htf_df: pd.DataFrame,
    htf_values: np.ndarray,
    use_completed_only: bool = True,
    additional_delay_bars: int = 0,
) -> np.ndarray:
    """
    Map higher-timeframe values back to lower-timeframe bars.

    CRITICAL: use_completed_only=True (default) ensures we only use the LAST
    COMPLETED HTF bar, not the current one still forming. This prevents look-ahead.
    Use additional_delay_bars for lagging HTF indicators that require extra
    future bars to confirm, such as Williams fractals.

    Args:
        prices: Primary (lower) timeframe DataFrame
        htf_df: Higher timeframe DataFrame (from get_htf_data)
        htf_values: numpy array of indicator values, same length as htf_df
        use_completed_only: If True, shift HTF values by 1 to avoid using unclosed bar
        additional_delay_bars: Extra completed HTF bars required before the value
            becomes tradable on the LTF. Example: Williams fractals need 2.

    Returns:
        numpy array of same length as prices, with HTF values forward-filled
    """
    if len(htf_df) != len(htf_values):
        raise ValueError(f"HTF length mismatch: df={len(htf_df)} values={len(htf_values)}")
    if len(prices) == 0:
        return np.array([], dtype=np.float64)
    if additional_delay_bars < 0:
        raise ValueError("additional_delay_bars must be >= 0")

    htf_open_times = pd.to_datetime(htf_df["open_time"])
    htf_deltas = htf_open_times.diff().dropna()
    if len(htf_deltas) == 0:
        inferred_duration = pd.Timedelta(0)
    else:
        inferred_duration = htf_deltas.mode().iloc[0]

    effective_times = htf_open_times.copy()
    if use_completed_only:
        # A HTF value becomes usable only when that HTF candle has CLOSED.
        effective_times = effective_times + inferred_duration
    if additional_delay_bars:
        effective_times = effective_times + inferred_duration * int(additional_delay_bars)

    htf_effective = pd.DataFrame(
        {
            "effective_time": effective_times,
            "value": np.asarray(htf_values, dtype=np.float64),
        }
    ).sort_values("effective_time", kind="stable")

    ltf_times = pd.DataFrame(
        {"open_time": pd.to_datetime(prices["open_time"])}
    ).sort_values("open_time", kind="stable")

    aligned = pd.merge_asof(
        ltf_times,
        htf_effective,
        left_on="open_time",
        right_on="effective_time",
        direction="backward",
        allow_exact_matches=True,
    )
    return aligned["value"].to_numpy(dtype=np.float64)


def compute_williams_fractals(
    high: np.ndarray,
    low: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Williams fractals on an HTF series.

    The raw fractal is indexed at the center bar, but it is only confirmed after
    two additional bars close. When aligning to a lower timeframe, call
    align_htf_to_ltf(..., additional_delay_bars=2).
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    n = len(high)

    bearish = np.zeros(n, dtype=np.float64)
    bullish = np.zeros(n, dtype=np.float64)

    for i in range(2, n - 2):
        if (
            high[i] >= high[i - 1]
            and high[i] >= high[i - 2]
            and high[i] >= high[i + 1]
            and high[i] >= high[i + 2]
        ):
            bearish[i] = 1.0
        if (
            low[i] <= low[i - 1]
            and low[i] <= low[i - 2]
            and low[i] <= low[i + 1]
            and low[i] <= low[i + 2]
        ):
            bullish[i] = 1.0

    return bearish, bullish


def compute_williams_fractal_levels(
    high: np.ndarray,
    low: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute raw Williams fractal price levels.

    Returns:
        bearish_levels: high price at bearish fractal bars, NaN elsewhere
        bullish_levels: low price at bullish fractal bars, NaN elsewhere

    These are raw center-bar levels. When mapped to a tradable timeframe they
    still require the 2-bar Williams confirmation delay.
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    bearish_flags, bullish_flags = compute_williams_fractals(high, low)

    bearish_levels = np.full(len(high), np.nan, dtype=np.float64)
    bullish_levels = np.full(len(low), np.nan, dtype=np.float64)

    bearish_idx = bearish_flags == 1.0
    bullish_idx = bullish_flags == 1.0
    bearish_levels[bearish_idx] = high[bearish_idx]
    bullish_levels[bullish_idx] = low[bullish_idx]

    return bearish_levels, bullish_levels


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
