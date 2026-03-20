#!/usr/bin/env python3
"""
strategy.py - Trend Momentum V4 Simplified
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "1h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Simplifying trend_momentum_v2 to reduce overfitting while maintaining
    core strengths. Recent complex additions (ADX, BB, multi-TF) hurt performance.
    
    Key changes from v2:
    - Simpler trend score: price position vs EMA stack (not weighted diffs)
    - RSI slope instead of absolute levels (momentum direction)
    - Volume ratio instead of percentile (cleaner normalization)
    - Fewer parameters to reduce overfitting risk
    - More conservative signal thresholds
    
    Core logic retained:
    - EMA stack for trend direction
    - RSI for momentum quality
    - Volume for confirmation
    - Volatility-based position sizing

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

name = "trend_momentum_v4_simplified"
timeframe = "1h"
leverage = 2.0  # Conservative leverage for risk-adjusted returns

# EMA periods for trend detection (simplified stack)
EMA_FAST = 12
EMA_MEDIUM = 26
EMA_SLOW = 50
EMA_MAJOR = 200

# RSI configuration
RSI_PERIOD = 14
RSI_MOMENTUM_THRESHOLD = 2.0  # Minimum RSI slope for confirmation

# Volume configuration
VOLUME_LOOKBACK = 20
VOLUME_RATIO_THRESHOLD = 1.2  # Volume must be 20% above average

# Trend scoring
TREND_MIN_SCORE = 0.2  # Minimum trend strength to trade

# Volatility configuration
ATR_PERIOD = 14
VOLATILITY_TARGET = 0.012  # Target hourly volatility
VOLATILITY_MIN = 0.003
VOLATILITY_MAX = 0.040

# Signal configuration
MIN_SIGNAL = 0.15
MAX_SIGNAL = 0.75
SMOOTHING_FACTOR = 0.6  # Exponential smoothing (lower = more smoothing)


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


def calculate_rsi_slope(rsi: np.ndarray, lookback: int = 3) -> np.ndarray:
    """
    Calculate RSI slope (momentum direction) using only past data.
    
    Args:
        rsi: Array of RSI values
        lookback: Number of periods for slope calculation
    
    Returns:
        Array of RSI slope values
    """
    n = len(rsi)
    slope = np.zeros(n, dtype=np.float64)
    
    if n < lookback + 1:
        return slope
    
    for i in range(lookback, n):
        slope[i] = rsi[i] - rsi[i - lookback]
    
    return slope


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


def calculate_volume_ratio(volume: np.ndarray, lookback: int = 20) -> np.ndarray:
    """
    Calculate volume ratio vs rolling average using only past data.
    
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
    volume_avg = volume_series.rolling(window=lookback, min_periods=lookback).mean().values
    
    for i in range(lookback, n):
        if volume_avg[i] > 0:
            volume_ratio[i] = volume[i] / volume_avg[i]
        else:
            volume_ratio[i] = 1.0
    
    volume_ratio = np.nan_to_num(volume_ratio, nan=1.0)
    
    return volume_ratio


def calculate_trend_score(close: float, ema_fast: float, ema_medium: float, 
                          ema_slow: float, ema_major: float) -> float:
    """
    Calculate trend score based on price position relative to EMA stack.
    Simpler than weighted EMA differences - focuses on alignment.
    
    Args:
        close: Current close price
        ema_fast: Fast EMA value
        ema_medium: Medium EMA value
        ema_slow: Slow EMA value
        ema_major: Major EMA value
    
    Returns:
        Trend score in range [-1, 1]
    """
    if close <= 0 or ema_major <= 0:
        return 0.0
    
    # Count how many EMAs price is above/below
    emas = [ema_fast, ema_medium, ema_slow, ema_major]
    above_count = sum(1 for ema in emas if close > ema)
    below_count = sum(1 for ema in emas if close < ema)
    
    # Check EMA stack alignment (fast > medium > slow > major for uptrend)
    stack_aligned_up = (ema_fast > ema_medium > ema_slow > ema_major)
    stack_aligned_down = (ema_fast < ema_medium < ema_slow < ema_major)
    
    # Base score from price position
    position_score = (above_count - below_count) / len(emas)
    
    # Amplify if stack is aligned
    if stack_aligned_up and position_score > 0:
        position_score *= 1.3
    elif stack_aligned_down and position_score < 0:
        position_score *= 1.3
    
    # Normalize to [-1, 1]
    trend_score = np.clip(position_score, -1.0, 1.0)
    
    return trend_score


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Trend Momentum V4 Simplified Strategy.
    
    Signal Logic:
    1. Price position relative to EMA stack for trend direction
    2. RSI slope for momentum confirmation
    3. Volume ratio for activity confirmation
    4. Volatility-based position sizing
    
    Entry Conditions:
    - LONG: Positive trend score + RSI slope positive + volume confirmed
    - SHORT: Negative trend score + RSI slope negative + volume confirmed
    
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
    
    # Handle NaN values
    close = np.nan_to_num(close, nan=0.0)
    high = np.nan_to_num(high, nan=0.0)
    low = np.nan_to_num(low, nan=0.0)
    volume = np.nan_to_num(volume, nan=0.0)
    
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
    rsi_slope = calculate_rsi_slope(rsi, lookback=3)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    volume_ratio = calculate_volume_ratio(volume, VOLUME_LOOKBACK)
    
    # Determine minimum valid index
    min_valid_index = max(
        EMA_MAJOR,
        RSI_PERIOD + 1,
        ATR_PERIOD + 1,
        VOLUME_LOOKBACK
    )
    
    # Track previous signal for smoothing
    prev_signal = 0.0
    
    # Generate signals
    for i in range(min_valid_index, n):
        # Skip invalid data
        if close[i] <= 0 or atr[i] <= 0:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Check volatility regime (avoid extreme volatility)
        atr_pct = atr[i] / close[i]
        if atr_pct < VOLATILITY_MIN or atr_pct > VOLATILITY_MAX:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Calculate trend score
        trend_score = calculate_trend_score(
            close[i], ema_fast[i], ema_medium[i], 
            ema_slow[i], ema_major[i]
        )
        
        # Skip weak trends
        if abs(trend_score) < TREND_MIN_SCORE:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # RSI momentum confirmation (slope direction matches trend)
        rsi_confirmed = False
        if trend_score > 0 and rsi_slope[i] > RSI_MOMENTUM_THRESHOLD:
            rsi_confirmed = True
        elif trend_score < 0 and rsi_slope[i] < -RSI_MOMENTUM_THRESHOLD:
            rsi_confirmed = True
        
        if not rsi_confirmed:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume_ratio[i] >= VOLUME_RATIO_THRESHOLD
        
        if not volume_confirmed:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Calculate base signal magnitude
        base_signal = trend_score
        
        # Apply volume boost if very strong
        if volume_ratio[i] >= 1.5:
            base_signal *= 1.1
        
        # Volatility-based position sizing (inverse relationship)
        vol_factor = VOLATILITY_TARGET / max(atr_pct, 0.001)
        vol_factor = np.clip(vol_factor, 0.5, 2.0)
        
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