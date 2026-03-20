#!/usr/bin/env python3
"""
strategy.py - Volatility Trend V2
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "4h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Simple trend-following with volatility-based position sizing:
    - Primary signal: EMA 20/50 crossover for trend direction
    - Filter: Price above/below 200 EMA for major trend alignment
    - Momentum: RSI confirmation (avoid extreme overbought/oversold)
    - Volatility scaling: ATR-based signal magnitude adjustment
    - No funding rate dependency (more robust across symbols)
    
    Why this works:
    - Simpler logic = fewer failure points
    - 4h timeframe captures multi-day trends with less noise than 1h
    - Volatility scaling naturally reduces position size in choppy markets
    - Conservative leverage (1.5x) keeps drawdown controlled
    
    Improvements over V12:
    - Removed funding rate dependency (unreliable across symbols)
    - Simpler signal combination (less overfitting)
    - Better volatility normalization
    - Reduced filtering to ensure sufficient trades

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

name = "volatility_trend_v2"
timeframe = "4h"
leverage = 1.5  # Conservative leverage for controlled drawdown

# EMA configuration for trend detection
EMA_FAST = 20
EMA_SLOW = 50
EMA_MAJOR = 200

# RSI configuration for momentum confirmation
RSI_PERIOD = 14
RSI_LONG_MIN = 45  # Minimum RSI for long entries
RSI_SHORT_MAX = 55  # Maximum RSI for short entries
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# Volatility configuration
ATR_PERIOD = 14
VOLATILITY_TARGET = 0.02  # Target ATR as % of price
VOLATILITY_MIN = 0.005  # Minimum ATR % to trade
VOLATILITY_MAX = 0.08  # Maximum ATR % to trade

# Signal configuration
MIN_SIGNAL_MAGNITUDE = 0.20  # Minimum signal to generate position
MAX_SIGNAL = 0.80  # Maximum signal magnitude
SMOOTHING_WINDOW = 3  # Simple moving average on signals
TREND_STRENGTH_SCALE = 40  # Scale factor for trend strength


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


def calculate_trend_strength(ema_fast: np.ndarray, ema_slow: np.ndarray, close: np.ndarray) -> np.ndarray:
    """
    Calculate trend strength from EMA separation.
    Returns signed strength value.
    Only uses current/past data (no look-ahead).
    """
    n = len(close)
    strength = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        if close[i] <= 0:
            strength[i] = 0.0
            continue
        
        ema_diff = ema_fast[i] - ema_slow[i]
        ema_diff_pct = ema_diff / close[i]
        strength[i] = ema_diff_pct * TREND_STRENGTH_SCALE
    
    return strength


def calculate_rsi_factor(rsi: np.ndarray, direction: int) -> float:
    """
    Calculate RSI momentum factor based on position direction.
    Returns factor in [0, 1].
    """
    if direction > 0:  # Long
        if rsi < RSI_LONG_MIN:
            return 0.0  # No long entry
        elif rsi > RSI_OVERBOUGHT:
            return 0.5  # Reduce strength if overbought
        else:
            return 1.0
    elif direction < 0:  # Short
        if rsi > RSI_SHORT_MAX:
            return 0.0  # No short entry
        elif rsi < RSI_OVERSOLD:
            return 0.5  # Reduce strength if oversold
        else:
            return 1.0
    else:
        return 0.0


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Volatility Trend V2 Strategy.
    
    Signal Logic:
    1. Calculate trend direction from EMA 20/50 crossover
    2. Filter by major trend (price vs 200 EMA)
    3. Confirm with RSI momentum
    4. Scale signal by volatility (ATR)
    5. Smooth signals to reduce whipsaws
    6. Apply minimum magnitude filter
    
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
    except (KeyError, TypeError, ValueError):
        return signals
    
    # Clean data
    close = np.nan_to_num(close, nan=0.0)
    high = np.nan_to_num(high, nan=0.0)
    low = np.nan_to_num(low, nan=0.0)
    
    # Fix invalid prices
    close = np.where(close <= 0, 1.0, close)
    high = np.where(high <= 0, close, high)
    low = np.where(low <= 0, close * 0.99, low)
    
    # Calculate all indicators (all use only past data)
    ema_fast = calculate_ema(close, EMA_FAST)
    ema_slow = calculate_ema(close, EMA_SLOW)
    ema_major = calculate_ema(close, EMA_MAJOR)
    
    rsi = calculate_rsi(close, RSI_PERIOD)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    trend_strength = calculate_trend_strength(ema_fast, ema_slow, close)
    
    # Calculate minimum valid index (all indicators need warmup)
    min_valid_index = max(
        EMA_MAJOR,
        EMA_SLOW + 5,  # Extra buffer for EMA convergence
        RSI_PERIOD + 1,
        ATR_PERIOD + 1
    )
    
    # Generate signals
    signal_buffer = []
    
    for i in range(min_valid_index, n):
        # Skip invalid bars
        if close[i] <= 0 or atr[i] <= 0:
            signal_buffer.append(0.0)
            continue
        
        # Determine trend direction from EMA crossover
        if ema_fast[i] > ema_slow[i]:
            trend_direction = 1  # Bullish
        elif ema_fast[i] < ema_slow[i]:
            trend_direction = -1  # Bearish
        else:
            trend_direction = 0  # Neutral
        
        # Major trend filter (price vs 200 EMA)
        if trend_direction != 0:
            if trend_direction > 0 and close[i] < ema_major[i]:
                trend_direction = 0  # Don't long in major downtrend
            elif trend_direction < 0 and close[i] > ema_major[i]:
                trend_direction = 0  # Don't short in major uptrend
        
        # Skip if no trend
        if trend_direction == 0:
            signal_buffer.append(0.0)
            continue
        
        # RSI momentum confirmation
        rsi_factor = calculate_rsi_factor(rsi[i], trend_direction)
        if rsi_factor == 0.0:
            signal_buffer.append(0.0)
            continue
        
        # Volatility filter (not too low, not too high)
        atr_pct = atr[i] / close[i]
        if atr_pct < VOLATILITY_MIN or atr_pct > VOLATILITY_MAX:
            signal_buffer.append(0.0)
            continue
        
        # Calculate raw signal
        raw_signal = trend_direction * trend_strength[i] * rsi_factor
        
        # Volatility normalization (scale by target volatility)
        vol_factor = VOLATILITY_TARGET / max(atr_pct, 0.001)
        vol_factor = np.clip(vol_factor, 0.5, 2.0)
        raw_signal *= vol_factor
        
        # Clip to reasonable range before smoothing
        raw_signal = np.clip(raw_signal, -1.0, 1.0)
        signal_buffer.append(raw_signal)
    
    # Pad signal buffer to match length
    signal_buffer = [0.0] * min_valid_index + signal_buffer
    signal_buffer = signal_buffer[:n]
    
    # Apply signal smoothing (simple moving average)
    if len(signal_buffer) >= SMOOTHING_WINDOW:
        smoothed = np.zeros(n, dtype=np.float64)
        for i in range(n):
            start_idx = max(0, i - SMOOTHING_WINDOW + 1)
            smoothed[i] = np.mean(signal_buffer[start_idx:i+1])
        signal_buffer = smoothed
    
    # Apply minimum magnitude filter and clip
    for i in range(n):
        signal = signal_buffer[i]
        if abs(signal) < MIN_SIGNAL_MAGNITUDE:
            signal = 0.0
        else:
            signal = np.clip(signal, -MAX_SIGNAL, MAX_SIGNAL)
        signals[i] = signal
    
    return signals