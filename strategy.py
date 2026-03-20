#!/usr/bin/env python3
"""
strategy.py - Multi-Timeframe Trend V1
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "1h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Testing multi-timeframe trend confirmation with simplified momentum.
    Building on trend_momentum_v2 (Sharpe=0.330) improvements:
    - Add longer-term trend filter (200 EMA as "4h proxy" on 1h data)
    - Simplify RSI logic (threshold-based instead of zone scoring)
    - Remove volume percentile (underperformed in #025)
    - Better volatility scaling based on recent ATR
    - Cleaner signal generation with less smoothing delay
    
    Key changes from v2:
    - 200 EMA must align with signal direction (major trend filter)
    - RSI simple threshold: >50 for long, <50 for short
    - Volatility scaling uses recent 14-bar ATR average
    - Reduced smoothing (0.5 vs 0.7) for faster entries
    - Minimum signal threshold increased (0.15 vs 0.10)

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

name = "multi_timeframe_trend_v1"
timeframe = "1h"
leverage = 2.0  # Conservative leverage for risk-adjusted returns

# EMA periods for trend detection
EMA_FAST = 12
EMA_MEDIUM = 26
EMA_SLOW = 50
EMA_MAJOR = 200  # Acts as multi-timeframe filter

# RSI configuration (simplified)
RSI_PERIOD = 14
RSI_LONG_THRESHOLD = 50
RSI_SHORT_THRESHOLD = 50

# Volatility configuration
ATR_PERIOD = 14
VOLATILITY_TARGET = 0.008  # Target hourly volatility
VOLATILITY_MIN = 0.002
VOLATILITY_MAX = 0.025

# Signal configuration
MIN_SIGNAL = 0.15
MAX_SIGNAL = 0.75
SMOOTHING_FACTOR = 0.5  # Less smoothing for faster entries

# Trend strength thresholds
MIN_TREND_STRENGTH = 0.002  # Minimum EMA separation as % of price


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


def calculate_trend_strength(close: float, ema_fast: float, ema_medium: float, 
                             ema_slow: float, ema_major: float) -> tuple:
    """
    Calculate trend strength and direction.
    
    Args:
        close: Current close price
        ema_fast: Fast EMA value
        ema_medium: Medium EMA value
        ema_slow: Slow EMA value
        ema_major: Major EMA value
    
    Returns:
        Tuple of (direction: -1/0/1, strength: float 0-1)
    """
    if close <= 0 or ema_major <= 0:
        return 0, 0.0
    
    # Calculate EMA separations as percentage of price
    fast_medium_sep = (ema_fast - ema_medium) / close
    medium_slow_sep = (ema_medium - ema_slow) / close
    slow_major_sep = (ema_slow - ema_major) / close
    close_major_sep = (close - ema_major) / close
    
    # Check alignment for bullish trend
    bullish_alignment = (
        fast_medium_sep > MIN_TREND_STRENGTH and
        medium_slow_sep > MIN_TREND_STRENGTH and
        slow_major_sep > MIN_TREND_STRENGTH and
        close_major_sep > MIN_TREND_STRENGTH
    )
    
    # Check alignment for bearish trend
    bearish_alignment = (
        fast_medium_sep < -MIN_TREND_STRENGTH and
        medium_slow_sep < -MIN_TREND_STRENGTH and
        slow_major_sep < -MIN_TREND_STRENGTH and
        close_major_sep < -MIN_TREND_STRENGTH
    )
    
    if bullish_alignment:
        # Calculate strength based on separation magnitude
        strength = min(1.0, (
            abs(fast_medium_sep) + 
            abs(medium_slow_sep) + 
            abs(slow_major_sep) + 
            abs(close_major_sep)
        ) / 0.04)  # Normalize by expected max separation
        return 1, strength
    elif bearish_alignment:
        strength = min(1.0, (
            abs(fast_medium_sep) + 
            abs(medium_slow_sep) + 
            abs(slow_major_sep) + 
            abs(close_major_sep)
        ) / 0.04)
        return -1, strength
    else:
        # Partial alignment - calculate net direction
        net_sep = fast_medium_sep + medium_slow_sep + slow_major_sep + close_major_sep
        direction = np.sign(net_sep)
        strength = min(1.0, abs(net_sep) / 0.02)
        return direction, strength


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Multi-Timeframe Trend V1 Strategy.
    
    Signal Logic:
    1. EMA stack alignment for trend direction
    2. 200 EMA as major trend filter (multi-timeframe proxy)
    3. RSI threshold confirmation (>50 for long, <50 for short)
    4. Volatility-based position sizing
    5. Light signal smoothing
    
    Entry Conditions:
    - LONG: Bullish EMA alignment + RSI > 50 + price > 200 EMA
    - SHORT: Bearish EMA alignment + RSI < 50 + price < 200 EMA
    
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
    except (KeyError, TypeError, ValueError):
        return signals
    
    # Handle NaN values
    close = np.nan_to_num(close, nan=0.0)
    high = np.nan_to_num(high, nan=0.0)
    low = np.nan_to_num(low, nan=0.0)
    
    # Ensure valid prices
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
    
    # Determine minimum valid index
    min_valid_index = max(
        EMA_MAJOR,
        RSI_PERIOD + 1,
        ATR_PERIOD + 1
    )
    
    # Track previous signal for smoothing
    prev_signal = 0.0
    
    # Generate signals
    for i in range(min_valid_index, n):
        # Skip invalid data
        if close[i] <= 0 or atr[i] <= 0 or ema_major[i] <= 0:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Check volatility regime (avoid extreme volatility)
        atr_pct = atr[i] / close[i]
        if atr_pct < VOLATILITY_MIN or atr_pct > VOLATILITY_MAX:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Calculate trend direction and strength
        trend_dir, trend_strength = calculate_trend_strength(
            close[i], ema_fast[i], ema_medium[i], 
            ema_slow[i], ema_major[i]
        )
        
        # Skip weak or no trend
        if trend_dir == 0 or trend_strength < 0.3:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Major trend filter: price must be on correct side of 200 EMA
        if trend_dir > 0 and close[i] < ema_major[i]:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        if trend_dir < 0 and close[i] > ema_major[i]:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # RSI confirmation
        if trend_dir > 0 and rsi[i] < RSI_LONG_THRESHOLD:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        if trend_dir < 0 and rsi[i] > RSI_SHORT_THRESHOLD:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Calculate base signal magnitude
        base_signal = trend_dir * trend_strength
        
        # Volatility-based position sizing (inverse relationship)
        vol_factor = VOLATILITY_TARGET / max(atr_pct, 0.001)
        vol_factor = np.clip(vol_factor, 0.5, 1.5)
        
        raw_signal = base_signal * vol_factor
        
        # Apply exponential smoothing to reduce whipsaws
        smoothed_signal = SMOOTHING_FACTOR * prev_signal + (1.0 - SMOOTHING_FACTOR) * raw_signal
        prev_signal = smoothed_signal
        
        # Apply thresholds
        if abs(smoothed_signal) < MIN_SIGNAL:
            smoothed_signal = 0.0
        
        # Clip to valid range
        signal = np.clip(smoothed_signal, -MAX_SIGNAL, MAX_SIGNAL)
        
        signals[i] = signal
    
    return signals