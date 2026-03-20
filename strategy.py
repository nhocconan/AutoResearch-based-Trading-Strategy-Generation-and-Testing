#!/usr/bin/env python3
"""
strategy.py - Multi-Timeframe Trend Volume V1
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "1h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Building on #015 success (Sharpe=0.316, Return=+556.4%), improving:
    - Multi-timeframe trend filter using slower EMAs as 4h/1d proxy
    - Volume breakout detection (spike above rolling average)
    - RSI momentum with divergence awareness
    - Better volatility regime filtering
    - Reduced signal churn with hysteresis
    
    Key improvements:
    - 200 EMA as major trend filter (4h/1d proxy on 1h data)
    - Volume spike detection (2x rolling average)
    - RSI momentum confirmation (not just zone)
    - Signal hysteresis to reduce flip-flopping
    - More conservative leverage (1.5 vs 2.0)

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

name = "multi_tf_trend_volume_v1"
timeframe = "1h"
leverage = 1.5  # Conservative leverage for better risk-adjusted returns

# EMA periods for multi-timeframe trend detection
EMA_FAST = 12      # Short-term momentum
EMA_MEDIUM = 26    # Medium-term trend
EMA_SLOW = 50      # Long-term trend
EMA_MAJOR = 200    # Multi-timeframe proxy (4h/1d on 1h data)

# RSI configuration
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
RSI_MOMENTUM_THRESHOLD = 55  # Minimum RSI for long momentum

# Volume configuration
VOLUME_LOOKBACK = 20
VOLUME_SPIKE_MULTIPLIER = 1.8  # Volume must be 1.8x average for breakout

# Trend scoring weights
WEIGHT_FAST = 0.35
WEIGHT_MEDIUM = 0.35
WEIGHT_SLOW = 0.30

# Volatility configuration
ATR_PERIOD = 14
VOLATILITY_TARGET = 0.008  # Target hourly volatility
VOLATILITY_MIN = 0.0015
VOLATILITY_MAX = 0.025

# Signal configuration
MIN_SIGNAL = 0.15
MAX_SIGNAL = 0.75
SMOOTHING_FACTOR = 0.75  # Higher smoothing to reduce churn
HYSTERESIS_THRESHOLD = 0.10  # Minimum change to flip signal direction


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


def calculate_volume_sma(volume: np.ndarray, lookback: int = 20) -> np.ndarray:
    """
    Calculate rolling volume SMA using only past data.
    
    Args:
        volume: Array of volume values
        lookback: Rolling window for SMA calculation
    
    Returns:
        Array of volume SMA values
    """
    n = len(volume)
    volume_sma = np.zeros(n, dtype=np.float64)
    
    if n < lookback:
        return volume_sma
    
    volume_series = pd.Series(volume)
    volume_sma_series = volume_series.rolling(window=lookback, min_periods=lookback).mean()
    
    volume_sma = np.nan_to_num(volume_sma_series.values, nan=0.0)
    
    return volume_sma


def calculate_trend_score(close: float, ema_fast: float, ema_medium: float, 
                          ema_slow: float, ema_major: float) -> float:
    """
    Calculate weighted trend score based on EMA stack alignment.
    Major EMA acts as multi-timeframe filter.
    
    Args:
        close: Current close price
        ema_fast: Fast EMA value
        ema_medium: Medium EMA value
        ema_slow: Slow EMA value
        ema_major: Major EMA value (multi-timeframe proxy)
    
    Returns:
        Trend score in range [-1, 1]
    """
    if close <= 0 or ema_major <= 0:
        return 0.0
    
    # Calculate individual trend components (normalized by price)
    fast_score = (ema_fast - ema_medium) / close
    medium_score = (ema_medium - ema_slow) / close
    slow_score = (ema_slow - ema_major) / close
    
    # Weight and combine
    trend_component = (
        WEIGHT_FAST * fast_score +
        WEIGHT_MEDIUM * medium_score +
        WEIGHT_SLOW * slow_score
    )
    
    # Major trend filter - only trade in direction of major trend
    major_direction = np.sign(close - ema_major)
    
    # Amplify signal when aligned with major trend, reduce when counter
    if major_direction > 0:
        trend_score = trend_component * 1.2  # Boost long signals in uptrend
    elif major_direction < 0:
        trend_score = trend_component * 1.2  # Boost short signals in downtrend
    else:
        trend_score = trend_component * 0.5  # Reduce signal in sideways market
    
    # Normalize to [-1, 1] range
    trend_score = np.clip(trend_score / 0.008, -1.0, 1.0)
    
    return trend_score


def calculate_rsi_momentum_score(rsi: float, trend_direction: int) -> float:
    """
    Calculate RSI momentum score based on trend direction.
    
    Args:
        rsi: Current RSI value
        trend_direction: +1 for long, -1 for short, 0 for neutral
    
    Returns:
        RSI momentum score in range [0, 1]
    """
    if trend_direction > 0:
        # Long bias: want RSI above momentum threshold but not overbought
        if rsi < RSI_MOMENTUM_THRESHOLD:
            return 0.2  # Weak momentum
        elif rsi < RSI_OVERBOUGHT:
            return 0.8  # Good momentum zone
        else:
            return 0.4  # Overbought, reduce confidence
    elif trend_direction < 0:
        # Short bias: want RSI below neutral but not oversold
        if rsi > (100 - RSI_MOMENTUM_THRESHOLD):
            return 0.2  # Weak momentum for short
        elif rsi > RSI_OVERSOLD:
            return 0.8  # Good momentum zone for short
        else:
            return 0.4  # Oversold, reduce confidence
    else:
        return 0.3  # Neutral trend


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Multi-Timeframe Trend Volume V1 Strategy.
    
    Signal Logic:
    1. Multi-timeframe trend filter (200 EMA as 4h/1d proxy)
    2. Weighted trend score from EMA stack alignment
    3. Volume breakout confirmation (spike above rolling average)
    4. RSI momentum confirmation
    5. Volatility-based position sizing
    6. Signal smoothing with hysteresis to reduce churn
    
    Entry Conditions:
    - LONG: Price > 200 EMA + positive trend + volume spike + RSI momentum
    - SHORT: Price < 200 EMA + negative trend + volume spike + RSI momentum
    
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
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    volume_sma = calculate_volume_sma(volume, VOLUME_LOOKBACK)
    
    # Determine minimum valid index
    min_valid_index = max(
        EMA_MAJOR,
        RSI_PERIOD + 1,
        ATR_PERIOD + 1,
        VOLUME_LOOKBACK
    )
    
    # Track previous signal for smoothing and hysteresis
    prev_signal = 0.0
    prev_direction = 0  # 0=neutral, 1=long, -1=short
    
    # Generate signals
    for i in range(min_valid_index, n):
        # Skip invalid data
        if close[i] <= 0 or atr[i] <= 0 or volume_sma[i] <= 0:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # Check volatility regime (avoid extreme volatility)
        atr_pct = atr[i] / close[i]
        if atr_pct < VOLATILITY_MIN or atr_pct > VOLATILITY_MAX:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # Multi-timeframe trend filter (200 EMA proxy)
        major_trend = np.sign(close[i] - ema_major[i])
        
        # Skip if no clear major trend
        if major_trend == 0:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # Calculate weighted trend score
        trend_score = calculate_trend_score(
            close[i], ema_fast[i], ema_medium[i], 
            ema_slow[i], ema_major[i]
        )
        
        # Skip weak trends
        if abs(trend_score) < 0.20:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # Check trend alignment with major trend
        if np.sign(trend_score) != major_trend:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # Volume breakout confirmation
        volume_ratio = volume[i] / volume_sma[i] if volume_sma[i] > 0 else 0
        volume_confirmed = volume_ratio >= VOLUME_SPIKE_MULTIPLIER
        
        # Require volume confirmation for new positions
        if prev_direction == 0 and not volume_confirmed:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # Calculate RSI momentum score
        trend_direction = 1 if trend_score > 0 else -1
        rsi_momentum = calculate_rsi_momentum_score(rsi[i], trend_direction)
        
        # Skip if RSI momentum is weak
        if rsi_momentum < 0.5:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # Determine base signal magnitude
        base_signal = trend_score * rsi_momentum
        
        # Apply volume confirmation boost
        if volume_confirmed:
            base_signal *= 1.2
        
        # Volatility-based position sizing (inverse relationship)
        vol_factor = VOLATILITY_TARGET / max(atr_pct, 0.001)
        vol_factor = np.clip(vol_factor, 0.5, 1.5)
        
        raw_signal = base_signal * vol_factor
        
        # Apply exponential smoothing to reduce whipsaws
        smoothed_signal = SMOOTHING_FACTOR * prev_signal + (1.0 - SMOOTHING_FACTOR) * raw_signal
        
        # Apply hysteresis to reduce direction flipping
        current_direction = np.sign(smoothed_signal)
        if current_direction != 0 and current_direction != prev_direction and prev_direction != 0:
            # Only flip if signal change exceeds hysteresis threshold
            if abs(smoothed_signal - prev_signal) < HYSTERESIS_THRESHOLD:
                smoothed_signal = prev_signal
        
        prev_signal = smoothed_signal
        prev_direction = np.sign(smoothed_signal)
        
        # Apply thresholds
        if abs(smoothed_signal) < MIN_SIGNAL:
            smoothed_signal = 0.0
            prev_direction = 0
        
        # Clip to valid range
        signal = np.clip(smoothed_signal, -MAX_SIGNAL, MAX_SIGNAL)
        
        signals[i] = signal
    
    return signals