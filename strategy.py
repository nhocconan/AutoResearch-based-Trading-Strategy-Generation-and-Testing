#!/usr/bin/env python3
"""
strategy.py - Simple Trend RSI V1
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "1h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    After recent experiments showed complex multi-indicator strategies failing,
    returning to simpler, proven concepts:
    - Clean EMA trend filter (fast > medium > slow for long, reverse for short)
    - RSI as momentum quality filter (not overbought for longs, not oversold for shorts)
    - Volatility-based position sizing (reduce size in high volatility)
    - Signal smoothing to reduce whipsaws
    - Cooling period after signal changes to reduce overtrading
    
    Key differences from trend_momentum_v2:
    - Simpler trend scoring (binary EMA alignment + magnitude)
    - RSI as filter not multiplier
    - Cleaner volatility adjustment
    - Added signal change cooldown

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

name = "simple_trend_rsi_v1"
timeframe = "1h"
leverage = 2.5  # Moderate leverage for better risk-adjusted returns

# EMA periods for trend detection
EMA_FAST = 12
EMA_MEDIUM = 26
EMA_SLOW = 50

# RSI configuration
RSI_PERIOD = 14
RSI_LONG_MIN = 45  # RSI must be above this for long entries
RSI_SHORT_MAX = 55  # RSI must be below this for short entries

# Volatility configuration
ATR_PERIOD = 14
VOLATILITY_TARGET = 0.012  # Target hourly volatility
VOLATILITY_MIN = 0.003
VOLATILITY_MAX = 0.040

# Signal configuration
MIN_SIGNAL = 0.15
MAX_SIGNAL = 0.85
SMOOTHING_FACTOR = 0.6  # Exponential smoothing factor (0-1)
SIGNAL_COOLDOWN = 3  # Bars to wait after signal direction change

# Trend strength threshold
TREND_STRENGTH_MIN = 0.002  # Minimum EMA spread as % of price


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
                             ema_slow: float) -> tuple:
    """
    Calculate trend direction and strength based on EMA stack.
    
    Args:
        close: Current close price
        ema_fast: Fast EMA value
        ema_medium: Medium EMA value
        ema_slow: Slow EMA value
    
    Returns:
        Tuple of (direction: -1/0/1, strength: 0-1)
    """
    if close <= 0 or ema_slow <= 0:
        return 0, 0.0
    
    # Calculate EMA spreads as percentage of price
    fast_medium_spread = (ema_fast - ema_medium) / close
    medium_slow_spread = (ema_medium - ema_slow) / close
    
    # Check for bullish alignment (fast > medium > slow)
    if fast_medium_spread > TREND_STRENGTH_MIN and medium_slow_spread > TREND_STRENGTH_MIN:
        direction = 1
        strength = min((fast_medium_spread + medium_slow_spread) / 0.02, 1.0)
    # Check for bearish alignment (fast < medium < slow)
    elif fast_medium_spread < -TREND_STRENGTH_MIN and medium_slow_spread < -TREND_STRENGTH_MIN:
        direction = -1
        strength = min(abs(fast_medium_spread + medium_slow_spread) / 0.02, 1.0)
    else:
        direction = 0
        strength = 0.0
    
    return direction, strength


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Simple Trend RSI V1 Strategy.
    
    Signal Logic:
    1. EMA stack alignment for trend direction
    2. RSI filter for momentum quality
    3. Volatility-based position sizing
    4. Signal smoothing with cooldown period
    
    Entry Conditions:
    - LONG: Bullish EMA stack + RSI > 45 + acceptable volatility
    - SHORT: Bearish EMA stack + RSI < 55 + acceptable volatility
    
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
    
    rsi = calculate_rsi(close, RSI_PERIOD)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Determine minimum valid index
    min_valid_index = max(
        EMA_SLOW,
        RSI_PERIOD + 1,
        ATR_PERIOD + 1
    )
    
    # Track previous signal for smoothing and cooldown
    prev_signal = 0.0
    prev_direction = 0  # 0=none, 1=long, -1=short
    cooldown_counter = 0
    
    # Generate signals
    for i in range(min_valid_index, n):
        # Skip invalid data
        if close[i] <= 0 or atr[i] <= 0:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            cooldown_counter = 0
            continue
        
        # Check volatility regime (avoid extreme volatility)
        atr_pct = atr[i] / close[i]
        if atr_pct < VOLATILITY_MIN or atr_pct > VOLATILITY_MAX:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            cooldown_counter = 0
            continue
        
        # Calculate trend direction and strength
        trend_direction, trend_strength = calculate_trend_strength(
            close[i], ema_fast[i], ema_medium[i], ema_slow[i]
        )
        
        # Skip weak or no trend
        if trend_direction == 0 or trend_strength < 0.3:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            cooldown_counter = 0
            continue
        
        # RSI filter based on trend direction
        rsi_ok = False
        if trend_direction > 0:
            # Long: RSI must not be too low (weak momentum)
            rsi_ok = rsi[i] >= RSI_LONG_MIN
        else:
            # Short: RSI must not be too high (strong momentum against us)
            rsi_ok = rsi[i] <= RSI_SHORT_MAX
        
        if not rsi_ok:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            cooldown_counter = 0
            continue
        
        # Calculate base signal magnitude
        base_signal = trend_direction * trend_strength
        
        # Volatility-based position sizing (inverse relationship)
        vol_factor = VOLATILITY_TARGET / max(atr_pct, 0.001)
        vol_factor = np.clip(vol_factor, 0.4, 2.5)
        
        raw_signal = base_signal * vol_factor
        
        # Check for signal direction change (cooldown logic)
        current_direction = 1 if raw_signal > 0 else (-1 if raw_signal < 0 else 0)
        
        if current_direction != 0 and current_direction != prev_direction and prev_direction != 0:
            # Direction changed, apply cooldown
            if cooldown_counter > 0:
                cooldown_counter -= 1
                # During cooldown, reduce signal magnitude
                raw_signal *= 0.3
            else:
                # First bar of new direction after cooldown
                cooldown_counter = SIGNAL_COOLDOWN
        elif current_direction != 0:
            # Same direction or new signal from flat
            cooldown_counter = 0
        
        # Apply exponential smoothing to reduce whipsaws
        smoothed_signal = SMOOTHING_FACTOR * prev_signal + (1.0 - SMOOTHING_FACTOR) * raw_signal
        
        # Apply thresholds
        if abs(smoothed_signal) < MIN_SIGNAL:
            smoothed_signal = 0.0
        
        # Clip to valid range
        signal = np.clip(smoothed_signal, -MAX_SIGNAL, MAX_SIGNAL)
        
        signals[i] = signal
        
        # Update tracking variables
        prev_signal = signal
        prev_direction = 1 if signal > 0 else (-1 if signal < 0 else 0)
    
    return signals