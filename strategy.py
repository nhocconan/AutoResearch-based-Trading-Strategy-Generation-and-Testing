#!/usr/bin/env python3
"""
strategy.py - Trend Volume Momentum V4 with Volatility Regime Filter
=====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "1h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Building on #010 success (Sharpe=0.330, Return=+40.7%), improving:
    - Cleaner signal generation with guaranteed [-1, 1] range
    - Volume momentum confirmation for institutional participation
    - Volatility regime filter to avoid chop and extreme volatility
    - Simplified trend detection with EMA stack alignment
    - Better position sizing based on ATR volatility
    
    Key improvements over #010:
    - Guaranteed signal bounds (no clipping errors)
    - Volume momentum Z-score for breakout confirmation
    - Bollinger Band width filter for volatility regime
    - ATR-based position sizing (smaller in high vol)
    - Cleaner code with fewer edge cases

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

name = "trend_volume_momentum_v4"
timeframe = "1h"
leverage = 2.0  # Conservative leverage for better risk-adjusted returns

# EMA periods for trend detection
EMA_FAST = 12
EMA_MEDIUM = 26
EMA_SLOW = 50
EMA_MAJOR = 200

# RSI configuration
RSI_PERIOD = 14
RSI_NEUTRAL = 50
RSI_STRONG = 60

# Volume configuration
VOLUME_LOOKBACK = 20
VOLUME_ZSCORE_THRESHOLD = 0.8  # Volume confirmation threshold

# Bollinger Band configuration for volatility regime
BB_PERIOD = 20
BB_STD = 2.0
BB_SQUEEZE_THRESHOLD = 0.012  # Band width below this = squeeze (avoid trading)
BB_EXPANSION_MIN = 0.018  # Band width above this = good volatility

# Volatility configuration
ATR_PERIOD = 14
VOLATILITY_TARGET = 0.010  # Target hourly volatility
VOLATILITY_MIN = 0.002
VOLATILITY_MAX = 0.035

# Signal configuration
MIN_SIGNAL_MAGNITUDE = 0.15  # Minimum signal to enter position
MAX_SIGNAL_MAGNITUDE = 0.85  # Maximum signal magnitude
SMOOTHING_FACTOR = 0.70  # Exponential smoothing factor (0-1)


# =============================================================================
# Helper Functions
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
    
    close_series = pd.Series(close)
    ema_values = close_series.ewm(span=period, adjust=False, min_periods=period).mean().values
    ema = np.nan_to_num(ema_values, nan=0.0)
    
    return ema


def calculate_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calculate Relative Strength Index using only past data.
    
    Args:
        close: Array of close prices
        period: RSI period
    
    Returns:
        Array of RSI values (0-100)
    """
    n = len(close)
    rsi = np.full(n, 50.0, dtype=np.float64)
    
    if n < period + 1:
        return rsi
    
    close_series = pd.Series(close)
    delta = close_series.diff()
    
    gains = delta.where(delta > 0, 0.0)
    losses = (-delta).where(delta < 0, 0.0)
    
    avg_gains = gains.ewm(com=period - 1, min_periods=period).mean()
    avg_losses = losses.ewm(com=period - 1, min_periods=period).mean()
    
    rs = avg_gains / avg_losses.replace(0, np.inf)
    rsi_series = 100.0 - (100.0 / (1.0 + rs))
    
    rsi = np.nan_to_num(rsi_series.values, nan=50.0)
    
    return rsi


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
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    
    tr_series = pd.Series(tr)
    atr_series = tr_series.ewm(span=period, adjust=False, min_periods=period).mean()
    
    atr = np.nan_to_num(atr_series.values, nan=0.0)
    
    return atr


def calculate_bollinger_bands(close: np.ndarray, period: int = 20, std_dev: float = 2.0):
    """
    Calculate Bollinger Bands using only past data.
    
    Args:
        close: Array of close prices
        period: BB period
        std_dev: Number of standard deviations
    
    Returns:
        Tuple of (upper, middle, lower, bandwidth) arrays
    """
    n = len(close)
    upper = np.zeros(n, dtype=np.float64)
    middle = np.zeros(n, dtype=np.float64)
    lower = np.zeros(n, dtype=np.float64)
    bandwidth = np.zeros(n, dtype=np.float64)
    
    if n < period:
        return upper, middle, lower, bandwidth
    
    close_series = pd.Series(close)
    middle_series = close_series.rolling(window=period, min_periods=period).mean()
    std_series = close_series.rolling(window=period, min_periods=period).std()
    
    upper = np.nan_to_num((middle_series + std_dev * std_series).values, nan=0.0)
    middle = np.nan_to_num(middle_series.values, nan=0.0)
    lower = np.nan_to_num((middle_series - std_dev * std_series).values, nan=0.0)
    
    # Bandwidth = (Upper - Lower) / Middle
    with np.errstate(divide='ignore', invalid='ignore'):
        bandwidth = np.where(middle > 0, (upper - lower) / middle, 0.0)
    
    return upper, middle, lower, bandwidth


def calculate_volume_zscore(volume: np.ndarray, lookback: int = 20) -> np.ndarray:
    """
    Calculate volume Z-score using rolling window.
    Only uses past volume data (no look-ahead).
    
    Args:
        volume: Array of volume values
        lookback: Rolling window for Z-score calculation
    
    Returns:
        Array of volume Z-scores
    """
    n = len(volume)
    volume_z = np.zeros(n, dtype=np.float64)
    
    if n < lookback:
        return volume_z
    
    volume_series = pd.Series(volume)
    rolling_mean = volume_series.rolling(window=lookback, min_periods=lookback).mean()
    rolling_std = volume_series.rolling(window=lookback, min_periods=lookback).std()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        volume_z = np.where(
            rolling_std > 0,
            (volume - rolling_mean.values) / rolling_std.values,
            0.0
        )
    
    volume_z = np.nan_to_num(volume_z, nan=0.0)
    
    return volume_z


def calculate_trend_strength(close: float, ema_fast: float, ema_medium: float, 
                             ema_slow: float, ema_major: float) -> float:
    """
    Calculate trend strength based on EMA stack alignment.
    Returns value in [-1, 1] where:
    - Positive = bullish alignment (fast > medium > slow > major)
    - Negative = bearish alignment (fast < medium < slow < major)
    - Near zero = choppy/no clear trend
    
    Args:
        close: Current close price
        ema_fast: Fast EMA value
        ema_medium: Medium EMA value
        ema_slow: Slow EMA value
        ema_major: Major EMA value
    
    Returns:
        Trend strength in range [-1, 1]
    """
    if close <= 0 or ema_major <= 0:
        return 0.0
    
    # Calculate normalized differences
    fast_med_diff = (ema_fast - ema_medium) / ema_medium if ema_medium > 0 else 0.0
    med_slow_diff = (ema_medium - ema_slow) / ema_slow if ema_slow > 0 else 0.0
    slow_major_diff = (ema_slow - ema_major) / ema_major if ema_major > 0 else 0.0
    close_major_diff = (close - ema_major) / ema_major if ema_major > 0 else 0.0
    
    # Check alignment (all same sign = strong trend)
    alignment_score = 0.0
    if fast_med_diff > 0 and med_slow_diff > 0 and slow_major_diff > 0:
        alignment_score = 1.0  # Perfect bullish alignment
    elif fast_med_diff < 0 and med_slow_diff < 0 and slow_major_diff < 0:
        alignment_score = -1.0  # Perfect bearish alignment
    else:
        # Partial alignment - calculate weighted average
        alignment_score = 0.4 * np.sign(fast_med_diff) + 0.3 * np.sign(med_slow_diff) + 0.3 * np.sign(slow_major_diff)
    
    # Combine with price position relative to major EMA
    trend_strength = 0.7 * alignment_score + 0.3 * np.sign(close_major_diff)
    
    # Scale to [-1, 1]
    trend_strength = np.clip(trend_strength, -1.0, 1.0)
    
    return trend_strength


def calculate_rsi_score(rsi: float, trend_direction: int) -> float:
    """
    Calculate RSI momentum score based on trend direction.
    Returns value in [0, 1] where higher = more favorable for the trend.
    
    Args:
        rsi: Current RSI value (0-100)
        trend_direction: +1 for long bias, -1 for short bias
    
    Returns:
        RSI score in range [0, 1]
    """
    if trend_direction > 0:
        # Long bias: want RSI > 50 (bullish momentum)
        if rsi >= RSI_STRONG:
            return 1.0  # Strong bullish momentum
        elif rsi >= RSI_NEUTRAL:
            return 0.7  # Moderate bullish
        elif rsi >= 40:
            return 0.4  # Weak/neutral
        else:
            return 0.2  # Bearish RSI (avoid long)
    else:
        # Short bias: want RSI < 50 (bearish momentum)
        if rsi <= (100 - RSI_STRONG):
            return 1.0  # Strong bearish momentum
        elif rsi <= (100 - RSI_NEUTRAL):
            return 0.7  # Moderate bearish
        elif rsi <= 60:
            return 0.4  # Weak/neutral
        else:
            return 0.2  # Bullish RSI (avoid short)


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Trend Volume Momentum V4 Strategy with Volatility Regime Filter.
    
    Signal Logic:
    1. EMA stack alignment for trend direction and strength
    2. RSI momentum scoring based on trend direction
    3. Volume Z-score for breakout confirmation
    4. Bollinger Band width filter (avoid squeeze zones)
    5. ATR-based position sizing (smaller in high volatility)
    6. Signal smoothing to reduce whipsaws
    
    Entry Conditions:
    - LONG: Bullish EMA stack + RSI > 50 + volume confirmation + BB expansion
    - SHORT: Bearish EMA stack + RSI < 50 + volume confirmation + BB expansion
    
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
    except (KeyError, TypeError, ValueError):
        return signals
    
    # Handle NaN values and ensure valid prices
    close = np.nan_to_num(close, nan=0.0)
    high = np.nan_to_num(high, nan=0.0)
    low = np.nan_to_num(low, nan=0.0)
    volume = np.nan_to_num(volume, nan=0.0)
    
    # Ensure positive prices
    close = np.where(close <= 0, 1.0, close)
    high = np.where(high <= 0, close, high)
    low = np.where(low <= 0, close * 0.99, low)
    
    # Calculate all indicators
    ema_fast = calculate_ema(close, EMA_FAST)
    ema_medium = calculate_ema(close, EMA_MEDIUM)
    ema_slow = calculate_ema(close, EMA_SLOW)
    ema_major = calculate_ema(close, EMA_MAJOR)
    
    rsi = calculate_rsi(close, RSI_PERIOD)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    volume_z = calculate_volume_zscore(volume, VOLUME_LOOKBACK)
    
    bb_upper, bb_middle, bb_lower, bb_width = calculate_bollinger_bands(
        close, BB_PERIOD, BB_STD
    )
    
    # Determine minimum valid index (all indicators need warmup period)
    min_valid_index = max(
        EMA_MAJOR,
        RSI_PERIOD + 1,
        ATR_PERIOD + 1,
        VOLUME_LOOKBACK,
        BB_PERIOD
    )
    
    # Track previous signal for smoothing
    prev_signal = 0.0
    
    # Generate signals
    for i in range(min_valid_index, n):
        # Skip invalid data
        if close[i] <= 0 or atr[i] <= 0 or bb_middle[i] <= 0:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Check Bollinger Band volatility regime (avoid squeeze zones)
        if bb_width[i] < BB_SQUEEZE_THRESHOLD:
            # Squeeze detected - stay flat or reduce position
            signals[i] = prev_signal * 0.3
            prev_signal = signals[i]
            continue
        
        # Check volatility regime (avoid extreme volatility)
        atr_pct = atr[i] / close[i]
        if atr_pct < VOLATILITY_MIN or atr_pct > VOLATILITY_MAX:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Calculate trend strength and direction
        trend_strength = calculate_trend_strength(
            close[i], ema_fast[i], ema_medium[i], 
            ema_slow[i], ema_major[i]
        )
        
        # Skip weak trends (choppy market)
        if abs(trend_strength) < 0.25:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Determine trend direction
        trend_direction = 1 if trend_strength > 0 else -1
        
        # Calculate RSI momentum score based on trend direction
        rsi_score = calculate_rsi_score(rsi[i], trend_direction)
        
        # Volume confirmation (Z-score based)
        volume_confirmed = volume_z[i] >= VOLUME_ZSCORE_THRESHOLD
        
        # Calculate base signal magnitude
        # Trend strength determines direction, RSI score determines conviction
        base_magnitude = abs(trend_strength) * rsi_score
        
        # Apply volume confirmation boost
        if volume_confirmed:
            base_magnitude *= 1.15
        
        # Volatility-based position sizing (inverse relationship)
        # Lower volatility = larger position, higher volatility = smaller position
        vol_factor = VOLATILITY_TARGET / max(atr_pct, 0.001)
        vol_factor = np.clip(vol_factor, 0.5, 2.0)
        
        # Calculate raw signal
        raw_signal = trend_direction * base_magnitude * vol_factor
        
        # Apply exponential smoothing to reduce whipsaws
        smoothed_signal = SMOOTHING_FACTOR * prev_signal + (1.0 - SMOOTHING_FACTOR) * raw_signal
        
        # Apply minimum signal threshold (avoid noise)
        if abs(smoothed_signal) < MIN_SIGNAL_MAGNITUDE:
            smoothed_signal = 0.0
        
        # Clip to valid range [-1, 1] - CRITICAL for engine compatibility
        signal = np.clip(smoothed_signal, -MAX_SIGNAL_MAGNITUDE, MAX_SIGNAL_MAGNITUDE)
        
        # Final safety check - ensure signal is in [-1, 1]
        signal = np.clip(signal, -1.0, 1.0)
        
        signals[i] = signal
        prev_signal = signal
    
    # Final safety check for entire array
    signals = np.clip(signals, -1.0, 1.0)
    
    return signals