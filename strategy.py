#!/usr/bin/env python3
"""
strategy.py - Trend Funding Simple V19
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "1h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Simplified trend-following with intelligent funding rate filtering:
    - Primary signal: EMA crossover (13/48) for faster trend capture
    - Filter: 200 EMA for directional bias only
    - Funding: Block trades at extremes, don't modify signal strength
    - Momentum: Simple price momentum confirmation
    - RSI: Only filter out extreme overbought/oversold
    
    Why this works:
    - Less complexity = more robust signals
    - Funding used as safety filter, not signal modifier
    - Faster EMAs capture trends earlier
    - Removed hysteresis and volatility normalization that dampened signals

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

name = "trend_funding_simple_v19"
timeframe = "1h"
leverage = 2.5  # Moderate leverage for better returns

# EMA configuration for trend detection
EMA_FAST = 13
EMA_SLOW = 48
EMA_MAJOR = 200

# RSI configuration for entry timing
RSI_PERIOD = 14
RSI_OVERBOUGHT = 75  # Don't long above this
RSI_OVERSOLD = 25    # Don't short below this

# Funding rate configuration
FUNDING_EXTREME_LONG = 0.0020   # Block longs if funding > 0.20% per 8hr
FUNDING_EXTREME_SHORT = -0.0020 # Block shorts if funding < -0.20% per 8hr
FUNDING_LOOKBACK = 50  # For calculating recent extremes

# Momentum configuration
MOMENTUM_PERIOD = 5  # Lookback for price momentum
MOMENTUM_THRESHOLD = 0.005  # Minimum momentum % to confirm trend

# Volume confirmation
VOLUME_LOOKBACK = 20
VOLUME_MIN_RATIO = 0.50  # Volume must be at least this % of average

# Signal configuration
MIN_SIGNAL_MAGNITUDE = 0.20  # Minimum signal to generate position
MAX_SIGNAL = 0.90  # Maximum signal magnitude
SMOOTHING_FACTOR = 0.30  # EMA smoothing for signals (0=none, 1=max)


# =============================================================================
# Helper Functions
# =============================================================================

def calculate_ema(close: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Exponential Moving Average using only past data.
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


def calculate_momentum(close: np.ndarray, period: int = 5) -> np.ndarray:
    """
    Calculate price momentum as percentage change over period.
    Only uses past data (no look-ahead).
    """
    n = len(close)
    momentum = np.zeros(n, dtype=np.float64)
    
    if n < period + 1:
        return momentum
    
    for i in range(period, n):
        if close[i - period] > 0:
            momentum[i] = (close[i] - close[i - period]) / close[i - period]
        else:
            momentum[i] = 0.0
    
    return momentum


def calculate_volume_ratio(volume: np.ndarray, lookback: int = 20) -> np.ndarray:
    """
    Calculate volume ratio vs rolling average.
    Only uses past volume data (no look-ahead).
    """
    n = len(volume)
    volume_ratio = np.ones(n, dtype=np.float64)
    
    if n < lookback:
        return volume_ratio
    
    volume_series = pd.Series(volume)
    rolling_avg = volume_series.rolling(window=lookback, min_periods=lookback).mean()
    
    volume_ratio = np.nan_to_num(volume_series.values / rolling_avg.values, nan=1.0)
    
    return volume_ratio


def calculate_funding_extremes(funding_rate: np.ndarray, lookback: int = 50) -> tuple:
    """
    Calculate rolling max/min of funding rate.
    Returns: (rolling_max, rolling_min)
    Only uses past funding rate data (no look-ahead).
    """
    n = len(funding_rate)
    rolling_max = np.zeros(n, dtype=np.float64)
    rolling_min = np.zeros(n, dtype=np.float64)
    
    if n < lookback:
        return rolling_max, rolling_min
    
    funding_series = pd.Series(funding_rate)
    rolling_max_series = funding_series.rolling(window=lookback, min_periods=lookback).max()
    rolling_min_series = funding_series.rolling(window=lookback, min_periods=lookback).min()
    
    rolling_max = np.nan_to_num(rolling_max_series.values, nan=0.0)
    rolling_min = np.nan_to_num(rolling_min_series.values, nan=0.0)
    
    return rolling_max, rolling_min


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Trend Funding Simple V19 Strategy.
    
    Signal Logic:
    1. Calculate trend signal from EMA crossover (13/48)
    2. Apply 200 EMA directional bias filter
    3. Check funding rate extremes (block trades at extremes)
    4. Confirm with momentum and RSI
    5. Apply volume filter
    6. Smooth signals slightly
    7. Apply minimum magnitude filter
    
    Args:
        prices: DataFrame with columns [open_time, open, high, low, close, volume, funding_rate, ...]
    
    Returns:
        np.ndarray of signals, same length as prices. Values in [-1, 1].
    """
    n = len(prices)
    signals = np.zeros(n, dtype=np.float64)
    
    # Extract price data with error handling
    try:
        close = prices["close"].values.astype(np.float64)
        high = prices["high"].values.astype(np.float64)
        low = prices["low"].values.astype(np.float64)
        volume = prices["volume"].values.astype(np.float64)
        
        try:
            funding_rate = prices["funding_rate"].values.astype(np.float64)
            funding_rate = np.nan_to_num(funding_rate, nan=0.0)
        except (KeyError, TypeError, ValueError):
            funding_rate = np.zeros(n, dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return signals
    
    # Clean data
    close = np.nan_to_num(close, nan=0.0)
    high = np.nan_to_num(high, nan=0.0)
    low = np.nan_to_num(low, nan=0.0)
    volume = np.nan_to_num(volume, nan=0.0)
    
    # Fix invalid prices
    close = np.where(close <= 0, 1.0, close)
    high = np.where(high <= 0, close, high)
    low = np.where(low <= 0, close * 0.99, low)
    
    # Calculate all indicators (all use only past data)
    ema_fast = calculate_ema(close, EMA_FAST)
    ema_slow = calculate_ema(close, EMA_SLOW)
    ema_major = calculate_ema(close, EMA_MAJOR)
    
    rsi = calculate_rsi(close, RSI_PERIOD)
    momentum = calculate_momentum(close, MOMENTUM_PERIOD)
    
    volume_ratio = calculate_volume_ratio(volume, VOLUME_LOOKBACK)
    funding_max, funding_min = calculate_funding_extremes(funding_rate, FUNDING_LOOKBACK)
    
    # Calculate minimum valid index (all indicators need warmup)
    min_valid_index = max(
        EMA_MAJOR,
        EMA_SLOW,
        RSI_PERIOD + 1,
        MOMENTUM_PERIOD + 1,
        VOLUME_LOOKBACK,
        FUNDING_LOOKBACK
    )
    
    # Generate signals
    prev_signal = 0.0
    
    for i in range(min_valid_index, n):
        # Skip invalid bars
        if close[i] <= 0 or ema_fast[i] <= 0 or ema_slow[i] <= 0:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Calculate EMA crossover signal
        ema_diff_pct = (ema_fast[i] - ema_slow[i]) / close[i]
        
        # Determine trend direction
        if ema_diff_pct > 0.0005:  # Fast EMA above slow
            trend_direction = 1.0
        elif ema_diff_pct < -0.0005:  # Fast EMA below slow
            trend_direction = -1.0
        else:
            trend_direction = 0.0
        
        # Skip if no clear trend
        if trend_direction == 0.0:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Apply 200 EMA directional bias filter
        # Only take longs if price > 200 EMA, shorts if price < 200 EMA
        if trend_direction > 0 and close[i] < ema_major[i]:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        elif trend_direction < 0 and close[i] > ema_major[i]:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Check funding rate extremes - BLOCK trades at extremes
        current_funding = funding_rate[i]
        
        # Block longs if funding is extremely positive (crowded long trade)
        if trend_direction > 0 and current_funding > FUNDING_EXTREME_LONG:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Block shorts if funding is extremely negative (crowded short trade)
        if trend_direction < 0 and current_funding < FUNDING_EXTREME_SHORT:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Check RSI filters
        if trend_direction > 0 and rsi[i] > RSI_OVERBOUGHT:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        elif trend_direction < 0 and rsi[i] < RSI_OVERSOLD:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Check momentum confirmation
        if trend_direction > 0 and momentum[i] < MOMENTUM_THRESHOLD:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        elif trend_direction < 0 and momentum[i] > -MOMENTUM_THRESHOLD:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Volume filter
        if volume_ratio[i] < VOLUME_MIN_RATIO:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Calculate signal strength based on EMA separation
        signal_strength = min(1.0, abs(ema_diff_pct) * 200)  # Scale to 0-1
        signal_strength = max(0.5, signal_strength)  # Minimum 0.5 strength
        
        # Apply direction
        raw_signal = trend_direction * signal_strength
        
        # Signal smoothing (EMA on signals)
        smoothed_signal = SMOOTHING_FACTOR * prev_signal + (1.0 - SMOOTHING_FACTOR) * raw_signal
        
        # Apply minimum magnitude filter
        if abs(smoothed_signal) < MIN_SIGNAL_MAGNITUDE:
            smoothed_signal = 0.0
        
        # Clip to max signal
        signal = np.clip(smoothed_signal, -MAX_SIGNAL, MAX_SIGNAL)
        
        signals[i] = signal
        prev_signal = signal
    
    return signals