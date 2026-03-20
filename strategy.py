#!/usr/bin/env python3
"""
strategy.py - Adaptive Trend-Mean Reversion Hybrid
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "1h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Building on #004's success (Sharpe=0.208), this strategy adds:
    - Bollinger Band position for volatility context
    - Adaptive RSI thresholds based on trend strength
    - Better volatility regime detection
    - Volume momentum confirmation
    
    Key improvements over #004:
    - BB position helps distinguish trend vs mean-reversion regimes
    - Dynamic RSI thresholds (tighter in strong trends, looser in weak)
    - Volatility regime filter (avoid trading in extreme volatility)
    - Smoother signal transitions

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

name = "adaptive_trend_mean_reversion"
timeframe = "1h"
leverage = 2.0  # Conservative leverage for better risk management

# Trend parameters
EMA_FAST = 12
EMA_SLOW = 26
EMA_CONFIRM = 50

# Bollinger Band parameters
BB_PERIOD = 20
BB_STD = 2.0

# RSI parameters
RSI_PERIOD = 14
RSI_LONG_MIN = 35
RSI_LONG_MAX = 70
RSI_SHORT_MIN = 30
RSI_SHORT_MAX = 65

# Volume parameters
VOLUME_LOOKBACK = 20
VOLUME_THRESHOLD = 1.1

# Volatility parameters
ATR_PERIOD = 14
VOLATILITY_TARGET = 0.012
VOLATILITY_MAX = 0.03  # Avoid trading in extreme volatility

# Signal parameters
MIN_SIGNAL = 0.12
MAX_SIGNAL = 0.75
TREND_STRENGTH_MIN = 0.0012


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


def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calculate Average True Range using only past data.
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


def calculate_bollinger_bands(close: np.ndarray, period: int = 20, std_dev: float = 2.0) -> tuple:
    """
    Calculate Bollinger Bands using only past data.
    Returns: (upper_band, middle_band, lower_band)
    """
    n = len(close)
    upper = np.zeros(n, dtype=np.float64)
    middle = np.zeros(n, dtype=np.float64)
    lower = np.zeros(n, dtype=np.float64)
    
    if n < period:
        return upper, middle, lower
    
    close_series = pd.Series(close)
    rolling_mean = close_series.rolling(window=period, min_periods=period).mean()
    rolling_std = close_series.rolling(window=period, min_periods=period).std()
    
    middle = np.nan_to_num(rolling_mean.values, nan=0.0)
    std_values = np.nan_to_num(rolling_std.values, nan=0.0)
    
    upper = middle + (std_dev * std_values)
    lower = middle - (std_dev * std_values)
    
    return upper, middle, lower


def calculate_volume_ratio(volume: np.ndarray, lookback: int = 20) -> np.ndarray:
    """
    Calculate volume ratio relative to rolling average.
    Only uses past volume data (no look-ahead).
    """
    n = len(volume)
    volume_ratio = np.ones(n, dtype=np.float64)
    
    if n < lookback:
        return volume_ratio
    
    volume_series = pd.Series(volume)
    rolling_avg = volume_series.rolling(window=lookback, min_periods=lookback).mean().values
    
    mask = rolling_avg > 0
    volume_ratio[mask] = volume[mask] / rolling_avg[mask]
    
    return volume_ratio


def calculate_bb_position(close: np.ndarray, upper: np.ndarray, lower: np.ndarray) -> np.ndarray:
    """
    Calculate position within Bollinger Bands.
    0 = at lower band, 0.5 = at middle, 1 = at upper band
    """
    n = len(close)
    bb_pos = np.full(n, 0.5, dtype=np.float64)
    
    for i in range(n):
        if upper[i] > lower[i]:
            bb_pos[i] = (close[i] - lower[i]) / (upper[i] - lower[i])
        else:
            bb_pos[i] = 0.5
    
    return bb_pos


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Adaptive Trend-Mean Reversion Hybrid Strategy.
    
    Signal Logic:
    1. Primary trend: EMA crossover for direction
    2. Volatility context: Bollinger Band position
    3. Momentum: RSI with adaptive thresholds
    4. Volume: Confirmation for breakouts
    5. Volatility filter: Avoid extreme volatility regimes
    
    Entry Conditions:
    - LONG: EMA_fast > EMA_slow > EMA_confirm + RSI 35-70 + BB position < 0.8
    - SHORT: EMA_fast < EMA_slow < EMA_confirm + RSI 30-65 + BB position > 0.2
    
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
    ema_slow = calculate_ema(close, EMA_SLOW)
    ema_confirm = calculate_ema(close, EMA_CONFIRM)
    
    rsi = calculate_rsi(close, RSI_PERIOD)
    
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    volume_ratio = calculate_volume_ratio(volume, VOLUME_LOOKBACK)
    
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close, BB_PERIOD, BB_STD)
    bb_position = calculate_bb_position(close, bb_upper, bb_lower)
    
    # Determine minimum valid index
    min_valid_index = max(
        EMA_CONFIRM,
        BB_PERIOD,
        RSI_PERIOD + 1,
        ATR_PERIOD + 1,
        VOLUME_LOOKBACK
    )
    
    # Generate signals
    for i in range(min_valid_index, n):
        # Skip invalid data
        if close[i] <= 0 or atr[i] <= 0:
            signals[i] = 0.0
            continue
        
        # Calculate trend strength (EMA spread normalized)
        ema_spread_fast_slow = (ema_fast[i] - ema_slow[i]) / close[i]
        ema_spread_slow_confirm = (ema_slow[i] - ema_confirm[i]) / close[i]
        
        # Trend direction and strength
        trend_bullish = (ema_fast[i] > ema_slow[i] > ema_confirm[i])
        trend_bearish = (ema_fast[i] < ema_slow[i] < ema_confirm[i])
        
        trend_strength = min(abs(ema_spread_fast_slow), abs(ema_spread_slow_confirm))
        
        # Filter: trend must be strong enough
        if trend_strength < TREND_STRENGTH_MIN:
            signals[i] = 0.0
            continue
        
        # Volatility regime filter - avoid extreme volatility
        atr_pct = atr[i] / close[i]
        if atr_pct > VOLATILITY_MAX:
            signals[i] = 0.0
            continue
        
        # Adaptive RSI thresholds based on trend strength
        # Stronger trend = wider RSI range allowed
        rsi_range_expansion = min(trend_strength / 0.005, 0.15)
        
        rsi_long_min_adj = RSI_LONG_MIN - (rsi_range_expansion * 10)
        rsi_long_max_adj = RSI_LONG_MAX + (rsi_range_expansion * 10)
        rsi_short_min_adj = RSI_SHORT_MIN - (rsi_range_expansion * 10)
        rsi_short_max_adj = RSI_SHORT_MAX + (rsi_range_expansion * 10)
        
        # RSI momentum filter with adaptive thresholds
        rsi_long_ok = rsi_long_min_adj <= rsi[i] <= rsi_long_max_adj
        rsi_short_ok = rsi_short_min_adj <= rsi[i] <= rsi_short_max_adj
        
        # Bollinger Band position filter
        # For longs: prefer not at upper band (overbought)
        # For shorts: prefer not at lower band (oversold)
        bb_long_ok = bb_position[i] < 0.85
        bb_short_ok = bb_position[i] > 0.15
        
        # Volume confirmation (optional boost)
        volume_confirmed = volume_ratio[i] >= VOLUME_THRESHOLD
        
        # Calculate signal
        raw_signal = 0.0
        signal_confidence = 0.0
        
        if trend_bullish and rsi_long_ok and bb_long_ok:
            # Long signal
            base_confidence = 0.5
            
            # Add confidence for stronger trend
            trend_factor = min(trend_strength / 0.006, 1.0)
            base_confidence += trend_factor * 0.3
            
            # Volume boost
            if volume_confirmed:
                base_confidence *= 1.12
            
            # BB position quality (prefer middle to lower half)
            bb_quality = 1.0
            if 0.3 <= bb_position[i] <= 0.6:
                bb_quality = 1.0
            elif 0.15 <= bb_position[i] < 0.3 or 0.6 < bb_position[i] <= 0.85:
                bb_quality = 0.92
            
            # RSI quality
            rsi_quality = 1.0
            if 45 <= rsi[i] <= 60:
                rsi_quality = 1.0
            elif 35 <= rsi[i] < 45 or 60 < rsi[i] <= 70:
                rsi_quality = 0.93
            
            signal_confidence = base_confidence * bb_quality * rsi_quality
            raw_signal = signal_confidence
            
        elif trend_bearish and rsi_short_ok and bb_short_ok:
            # Short signal
            base_confidence = 0.5
            
            trend_factor = min(trend_strength / 0.006, 1.0)
            base_confidence += trend_factor * 0.3
            
            if volume_confirmed:
                base_confidence *= 1.12
            
            # BB position quality (prefer middle to upper half)
            bb_quality = 1.0
            if 0.4 <= bb_position[i] <= 0.7:
                bb_quality = 1.0
            elif 0.15 <= bb_position[i] < 0.4 or 0.7 < bb_position[i] <= 0.85:
                bb_quality = 0.92
            
            # RSI quality
            rsi_quality = 1.0
            if 40 <= rsi[i] <= 55:
                rsi_quality = 1.0
            elif 30 <= rsi[i] < 40 or 55 < rsi[i] <= 65:
                rsi_quality = 0.93
            
            signal_confidence = base_confidence * bb_quality * rsi_quality
            raw_signal = -signal_confidence
        
        # Apply volatility adjustment
        vol_factor = 1.0
        if atr_pct > 0:
            vol_factor = min(1.4, VOLATILITY_TARGET / max(atr_pct, 0.001))
        
        signal = raw_signal * vol_factor
        
        # Apply thresholds
        if abs(signal) < MIN_SIGNAL:
            signal = 0.0
        
        signal = np.clip(signal, -MAX_SIGNAL, MAX_SIGNAL)
        
        signals[i] = signal
    
    return signals