#!/usr/bin/env python3
"""
strategy.py - Pure Trend Follower V22
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

Strategy Hypothesis:
    Pure trend-following without conflicting overlays:
    - Primary signal: EMA crossover (12/26) for trend direction
    - Filter: Price above/below 200 EMA for major trend validation
    - Entry timing: RSI momentum (avoid extreme overbought/oversold)
    - Volume confirmation: Ensure sufficient liquidity
    - NO funding rate overlay (was conflicting with trend signals)
    - NO volatility normalization (was distorting signal magnitudes)
    - Minimal smoothing (reduce entry delay)
    
    Why this works:
    - Crypto has strong trend persistence
    - Fewer conflicting signals = cleaner execution
    - Simpler = more robust across market regimes
    - Focus on quality trend captures, not frequency

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

name = "pure_trend_follower_v22"
timeframe = "1h"
leverage = 2.5  # Moderate leverage for trend following

# EMA configuration - faster response than v12
EMA_FAST = 12
EMA_SLOW = 26
EMA_MAJOR = 200

# RSI configuration for entry timing
RSI_PERIOD = 14
RSI_LONG_MIN = 40  # RSI must be above this for longs
RSI_SHORT_MAX = 60  # RSI must be below this for shorts
RSI_OVERBOUGHT = 75  # Avoid entering when overbought
RSI_OVERSOLD = 25  # Avoid entering when oversold

# Volume configuration
VOLUME_LOOKBACK = 20
VOLUME_MIN_RATIO = 0.5  # Volume must be at least this % of average

# Signal configuration - simplified from v12
MIN_SIGNAL = 0.3  # Minimum signal magnitude to trade
MAX_SIGNAL = 0.9  # Maximum signal magnitude
SMOOTHING = 0.3  # Light smoothing (was 0.50 in v12)


# =============================================================================
# Helper Functions
# =============================================================================

def calculate_ema(close: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Exponential Moving Average using only past data.
    """
    if len(close) < period:
        return np.zeros(len(close), dtype=np.float64)
    
    series = pd.Series(close)
    ema = series.ewm(span=period, adjust=False, min_periods=period).mean()
    return np.nan_to_num(ema.values, nan=0.0)


def calculate_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calculate Relative Strength Index using only past data.
    """
    if len(close) < period + 1:
        return np.full(len(close), 50.0, dtype=np.float64)
    
    series = pd.Series(close)
    delta = series.diff()
    
    gains = delta.where(delta > 0, 0.0)
    losses = (-delta).where(delta < 0, 0.0)
    
    avg_gains = gains.ewm(com=period - 1, min_periods=period).mean()
    avg_losses = losses.ewm(com=period - 1, min_periods=period).mean()
    
    rs = avg_gains / avg_losses.replace(0, np.inf)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return np.nan_to_num(rsi.values, nan=50.0)


def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calculate Average True Range using only past data.
    """
    if len(close) < period + 1:
        return np.zeros(len(close), dtype=np.float64)
    
    tr = np.zeros(len(close), dtype=np.float64)
    tr[0] = high[0] - low[0]
    
    for i in range(1, len(close)):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    
    tr_series = pd.Series(tr)
    atr = tr_series.ewm(span=period, adjust=False, min_periods=period).mean()
    
    return np.nan_to_num(atr.values, nan=0.0)


def calculate_volume_ratio(volume: np.ndarray, lookback: int = 20) -> np.ndarray:
    """
    Calculate volume ratio vs rolling average.
    Only uses past volume data (no look-ahead).
    """
    if len(volume) < lookback:
        return np.ones(len(volume), dtype=np.float64)
    
    series = pd.Series(volume)
    rolling_avg = series.rolling(window=lookback, min_periods=lookback).mean()
    
    return np.nan_to_num(series.values / rolling_avg.values, nan=1.0)


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Pure Trend Follower V22 Strategy.
    
    Signal Logic:
    1. EMA crossover (12/26) for trend direction
    2. 200 EMA filter for major trend validation
    3. RSI entry timing confirmation
    4. Volume liquidity filter
    5. Light smoothing for cleaner signals
    
    Args:
        prices: DataFrame with columns [open_time, open, high, low, close, volume, ...]
    
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
    atr = calculate_atr(high, low, close, 14)
    volume_ratio = calculate_volume_ratio(volume, VOLUME_LOOKBACK)
    
    # Minimum valid index (all indicators need warmup)
    min_valid = max(
        EMA_MAJOR,
        EMA_SLOW,
        RSI_PERIOD + 1,
        15,  # ATR warmup
        VOLUME_LOOKBACK
    )
    
    prev_signal = 0.0
    
    for i in range(min_valid, n):
        # Skip invalid bars
        if close[i] <= 0 or ema_major[i] <= 0:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Volume filter (ensure sufficient liquidity)
        if volume_ratio[i] < VOLUME_MIN_RATIO:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # EMA crossover signal
        ema_diff = (ema_fast[i] - ema_slow[i]) / close[i]
        ema_direction = np.sign(ema_diff)
        
        # Skip if no clear trend
        if ema_direction == 0:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Trend strength (scaled)
        trend_strength = min(abs(ema_diff) * 100, 1.0)
        
        # Major trend filter - only trade with the major trend
        major_direction = np.sign(close[i] - ema_major[i])
        
        if ema_direction != major_direction:
            # Conflicting signals - skip trade
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # RSI confirmation for entry timing
        rsi_ok = False
        if ema_direction > 0:
            # Long: RSI should be above minimum but not overbought
            if RSI_LONG_MIN < rsi[i] < RSI_OVERBOUGHT:
                rsi_ok = True
        elif ema_direction < 0:
            # Short: RSI should be below maximum but not oversold
            if RSI_SHORT_MAX > rsi[i] > RSI_OVERSOLD:
                rsi_ok = True
        
        if not rsi_ok:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Calculate raw signal
        raw_signal = ema_direction * trend_strength
        
        # Light smoothing to reduce noise
        smoothed = SMOOTHING * prev_signal + (1.0 - SMOOTHING) * raw_signal
        
        # Minimum magnitude filter - don't trade weak signals
        if abs(smoothed) < MIN_SIGNAL:
            smoothed = 0.0
        
        # Clip to maximum signal
        signal = np.clip(smoothed, -MAX_SIGNAL, MAX_SIGNAL)
        
        signals[i] = signal
        prev_signal = signal
    
    return signals