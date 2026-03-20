#!/usr/bin/env python3
"""
strategy.py - Multi-Timeframe Trend Following with Volume Confirmation
=======================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "1h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Trend-following strategy with multi-timeframe confirmation and volume filter.
    - Use 20/50 EMA crossover for trend direction
    - Volume spike confirms breakout validity
    - ATR-based position sizing for volatility adjustment
    - Avoid trading during low volume periods

Look-Ahead Safety:
    - All rolling calculations use only past data (min_periods respected)
    - No .shift(-n) or future index access
    - Signal at bar t uses only prices.iloc[:t+1]
"""

import numpy as np
import pandas as pd

# =============================================================================
# Strategy Configuration
# =============================================================================

name = "multi_tf_trend_volume"
timeframe = "1h"
leverage = 2.5  # Moderate leverage for trend following

# Strategy parameters
EMA_FAST = 20             # Fast EMA period
EMA_SLOW = 50             # Slow EMA period
VOLUME_LOOKBACK = 20      # Lookback for volume average
VOLUME_THRESHOLD = 1.5    # Volume spike multiplier
ATR_PERIOD = 14           # ATR calculation period
TREND_STRENGTH_WINDOW = 10  # Window for trend strength calculation
MIN_SIGNAL = 0.3          # Minimum signal magnitude to trade


# =============================================================================
# Signal Generation
# =============================================================================

def calculate_ema(close: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Exponential Moving Average using only past data.
    
    Args:
        close: Array of close prices
        period: EMA period
    
    Returns:
        Array of EMA values
    """
    n = len(close)
    ema = np.zeros(n, dtype=np.float64)
    
    if n < period:
        return ema
    
    # Initialize with SMA
    ema[period - 1] = np.mean(close[:period])
    
    # Calculate EMA multiplier
    multiplier = 2.0 / (period + 1)
    
    # Calculate EMA for remaining periods
    for i in range(period, n):
        ema[i] = (close[i] - ema[i-1]) * multiplier + ema[i-1]
    
    return ema


def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calculate Average True Range using only past data.
    
    Args:
        high: Array of high prices
        low: Array of low prices
        close: Array of close prices
        period: ATR period
    
    Returns:
        Array of ATR values
    """
    n = len(close)
    atr = np.zeros(n, dtype=np.float64)
    
    if n < period + 1:
        return atr
    
    # Calculate True Range
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    
    # Initialize ATR with SMA of TR
    atr[period - 1] = np.mean(tr[:period])
    
    # Calculate ATR using Wilder's smoothing
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    
    return atr


def calculate_volume_ratio(volume: np.ndarray, lookback: int = 20) -> np.ndarray:
    """
    Calculate volume ratio relative to rolling average.
    Only uses past volume data (no look-ahead).
    
    Args:
        volume: Array of volume values
        lookback: Rolling window for average calculation
    
    Returns:
        Array of volume ratios
    """
    n = len(volume)
    volume_ratio = np.ones(n, dtype=np.float64)
    
    if n < lookback:
        return volume_ratio
    
    volume_series = pd.Series(volume)
    rolling_avg = volume_series.rolling(window=lookback, min_periods=lookback).mean().values
    
    # Avoid division by zero
    mask = rolling_avg > 0
    volume_ratio[mask] = volume[mask] / rolling_avg[mask]
    
    return volume_ratio


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Multi-Timeframe Trend Following Strategy with Volume Confirmation.
    
    Signal Logic:
    1. Calculate fast/slow EMA for trend direction
    2. Calculate ATR for volatility adjustment
    3. Calculate volume ratio for confirmation
    4. Generate signals based on EMA crossover and volume
    
    Entry Conditions:
    - LONG: Fast EMA > Slow EMA AND volume ratio > threshold
    - SHORT: Fast EMA < Slow EMA AND volume ratio > threshold
    
    Args:
        prices: DataFrame with columns [open_time, open, high, low, close, volume, ...]
    
    Returns:
        np.ndarray of signals, same length as prices. Values in [-1, 1].
    """
    n = len(prices)
    signals = np.zeros(n, dtype=np.float64)
    
    # Extract required columns with safety checks
    try:
        close = prices["close"].values.astype(np.float64)
        high = prices["high"].values.astype(np.float64)
        low = prices["low"].values.astype(np.float64)
        volume = prices["volume"].values.astype(np.float64)
    except (KeyError, TypeError, ValueError) as e:
        # Return zeros if required columns missing
        return signals
    
    # Handle any NaN values in price data
    close = np.nan_to_num(close, nan=0.0)
    high = np.nan_to_num(high, nan=0.0)
    low = np.nan_to_num(low, nan=0.0)
    volume = np.nan_to_num(volume, nan=0.0)
    
    # Ensure no zero or negative prices
    close = np.where(close <= 0, 1.0, close)
    high = np.where(high <= 0, close, high)
    low = np.where(low <= 0, close * 0.99, low)
    
    # Calculate EMAs
    ema_fast = calculate_ema(close, EMA_FAST)
    ema_slow = calculate_ema(close, EMA_SLOW)
    
    # Calculate ATR for volatility adjustment
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Calculate volume ratio
    volume_ratio = calculate_volume_ratio(volume, VOLUME_LOOKBACK)
    
    # Calculate trend strength (EMA spread normalized by price)
    ema_spread = (ema_fast - ema_slow) / close
    ema_spread = np.nan_to_num(ema_spread, nan=0.0)
    
    # Calculate rolling trend strength for smoothing
    trend_strength_series = pd.Series(ema_spread).rolling(
        window=TREND_STRENGTH_WINDOW, 
        min_periods=TREND_STRENGTH_WINDOW
    ).mean().values
    trend_strength = np.nan_to_num(trend_strength_series, nan=0.0)
    
    # Determine minimum valid index
    min_valid_index = max(EMA_SLOW, ATR_PERIOD + 1, VOLUME_LOOKBACK, TREND_STRENGTH_WINDOW)
    
    # Generate signals
    for i in range(min_valid_index, n):
        # Skip if any required data is invalid
        if close[i] <= 0 or atr[i] <= 0:
            signals[i] = 0.0
            continue
        
        # Trend direction from EMA crossover
        ema_bullish = ema_fast[i] > ema_slow[i]
        ema_bearish = ema_fast[i] < ema_slow[i]
        
        # Volume confirmation
        volume_confirmed = volume_ratio[i] >= VOLUME_THRESHOLD
        
        # Trend strength (normalized)
        trend_mag = min(abs(trend_strength[i]) * 100, 1.0)  # Cap at 1.0
        
        # Volatility adjustment (reduce position in high volatility)
        # Normalize ATR by price to get percentage volatility
        atr_pct = atr[i] / close[i]
        vol_factor = 1.0
        if atr_pct > 0:
            # Typical 1h ATR% is 0.5-2%, scale inversely
            vol_factor = min(1.0, 0.015 / max(atr_pct, 0.001))
        
        # Base signal from trend direction
        raw_signal = 0.0
        if ema_bullish:
            raw_signal = trend_mag
        elif ema_bearish:
            raw_signal = -trend_mag
        
        # Apply volume confirmation (reduce signal if volume low)
        if not volume_confirmed:
            raw_signal *= 0.5
        
        # Apply volatility adjustment
        signal = raw_signal * vol_factor
        
        # Apply minimum signal threshold
        if abs(signal) < MIN_SIGNAL:
            signal = 0.0
        
        # Clip to [-1, 1]
        signal = np.clip(signal, -1.0, 1.0)
        
        signals[i] = signal
    
    return signals