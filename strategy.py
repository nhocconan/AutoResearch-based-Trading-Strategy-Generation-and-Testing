#!/usr/bin/env python3
"""
strategy.py - Trend Momentum V3 with Volume Confirmation
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "1h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Building on trend_momentum_v2 (Sharpe=0.330), improving:
    - Simplified EMA stack (3 EMAs instead of 4) for cleaner signals
    - Volume ratio vs MA instead of percentile (more responsive)
    - Cleaner RSI momentum integration (threshold-based)
    - Reduced signal smoothing lag (0.5 vs 0.7)
    - Better volatility regime filtering
    
    Key improvements over V2:
    - Fewer EMAs = less lag, clearer trend signals
    - Volume ratio more responsive than percentile ranking
    - RSI thresholds simpler than zone scoring
    - Less smoothing = faster reaction to trend changes
    - Tighter volatility bounds for better risk control

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

name = "trend_momentum_v3"
timeframe = "1h"
leverage = 2.0

# EMA periods for trend detection (simplified from V2)
EMA_FAST = 12
EMA_MEDIUM = 26
EMA_SLOW = 50

# RSI configuration
RSI_PERIOD = 14
RSI_BULLISH_THRESHOLD = 55
RSI_BEARISH_THRESHOLD = 45

# Volume configuration (simpler than percentile)
VOLUME_MA_PERIOD = 20
VOLUME_RATIO_THRESHOLD = 1.2  # Volume must be 20% above average

# Trend scoring
TREND_MIN_STRENGTH = 0.002  # Minimum trend strength to trade

# Volatility configuration
ATR_PERIOD = 14
VOLATILITY_TARGET = 0.008
VOLATILITY_MIN = 0.003
VOLATILITY_MAX = 0.025

# Signal configuration
MIN_SIGNAL = 0.15
MAX_SIGNAL = 0.75
SMOOTHING_FACTOR = 0.5  # Reduced from 0.7 for less lag


# =============================================================================
# Helper Functions
# =============================================================================

def calculate_ema(close: np.ndarray, period: int) -> np.ndarray:
    """Calculate EMA using only past data."""
    n = len(close)
    ema = np.zeros(n, dtype=np.float64)
    
    if n < period:
        return ema
    
    close_series = pd.Series(close)
    ema_values = close_series.ewm(span=period, adjust=False, min_periods=period).mean().values
    ema = np.nan_to_num(ema_values, nan=0.0)
    
    return ema


def calculate_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate RSI using only past data."""
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
    """Calculate ATR using only past data."""
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


def calculate_volume_ratio(volume: np.ndarray, period: int = 20) -> np.ndarray:
    """Calculate volume ratio vs moving average."""
    n = len(volume)
    volume_ratio = np.ones(n, dtype=np.float64)
    
    if n < period:
        return volume_ratio
    
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=period, min_periods=period).mean().values
    
    for i in range(period, n):
        if volume_ma[i] > 0:
            volume_ratio[i] = volume[i] / volume_ma[i]
        else:
            volume_ratio[i] = 1.0
    
    volume_ratio = np.nan_to_num(volume_ratio, nan=1.0)
    
    return volume_ratio


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Trend Momentum V3 Strategy with Volume Confirmation.
    
    Signal Logic:
    1. Trend direction from EMA stack alignment
    2. RSI momentum confirmation
    3. Volume ratio for breakout confirmation
    4. Volatility-based position sizing
    
    Entry Conditions:
    - LONG: EMA fast > medium > slow + RSI > 55 + volume ratio > 1.2
    - SHORT: EMA fast < medium < slow + RSI < 45 + volume ratio > 1.2
    
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
    
    rsi = calculate_rsi(close, RSI_PERIOD)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    volume_ratio = calculate_volume_ratio(volume, VOLUME_MA_PERIOD)
    
    # Determine minimum valid index
    min_valid_index = max(
        EMA_SLOW,
        RSI_PERIOD + 1,
        ATR_PERIOD + 1,
        VOLUME_MA_PERIOD
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
        
        # Check volatility regime
        atr_pct = atr[i] / close[i]
        if atr_pct < VOLATILITY_MIN or atr_pct > VOLATILITY_MAX:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Calculate trend strength and direction
        trend_strength = (ema_fast[i] - ema_slow[i]) / close[i]
        trend_direction = np.sign(trend_strength)
        
        # Check EMA stack alignment
        if trend_direction > 0:
            # Bullish alignment: fast > medium > slow
            ema_aligned = (ema_fast[i] > ema_medium[i]) and (ema_medium[i] > ema_slow[i])
        else:
            # Bearish alignment: fast < medium < slow
            ema_aligned = (ema_fast[i] < ema_medium[i]) and (ema_medium[i] < ema_slow[i])
        
        if not ema_aligned or abs(trend_strength) < TREND_MIN_STRENGTH:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # RSI momentum confirmation
        if trend_direction > 0:
            rsi_confirmed = rsi[i] > RSI_BULLISH_THRESHOLD
            rsi_factor = (rsi[i] - RSI_BULLISH_THRESHOLD) / (100 - RSI_BULLISH_THRESHOLD)
        else:
            rsi_confirmed = rsi[i] < RSI_BEARISH_THRESHOLD
            rsi_factor = (RSI_BEARISH_THRESHOLD - rsi[i]) / RSI_BEARISH_THRESHOLD
        
        if not rsi_confirmed:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume_ratio[i] > VOLUME_RATIO_THRESHOLD
        volume_factor = min(volume_ratio[i] / VOLUME_RATIO_THRESHOLD, 1.5)
        
        if not volume_confirmed:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Calculate base signal
        base_signal = trend_direction * abs(trend_strength) * 100  # Scale to reasonable range
        base_signal = base_signal * rsi_factor * volume_factor
        
        # Volatility-based position sizing
        vol_factor = VOLATILITY_TARGET / max(atr_pct, 0.001)
        vol_factor = np.clip(vol_factor, 0.5, 2.0)
        
        raw_signal = base_signal * vol_factor
        
        # Apply smoothing (reduced lag from V2)
        smoothed_signal = SMOOTHING_FACTOR * prev_signal + (1.0 - SMOOTHING_FACTOR) * raw_signal
        prev_signal = smoothed_signal
        
        # Apply thresholds
        if abs(smoothed_signal) < MIN_SIGNAL:
            smoothed_signal = 0.0
        
        # Clip to valid range
        signal = np.clip(smoothed_signal, -MAX_SIGNAL, MAX_SIGNAL)
        
        signals[i] = signal
    
    return signals