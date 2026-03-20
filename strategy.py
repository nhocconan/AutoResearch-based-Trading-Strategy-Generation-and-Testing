#!/usr/bin/env python3
"""
strategy.py - Trend Momentum V3 Simplified
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "1h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Learning from #022 failure (Sharpe=-0.890), simplifying the approach:
    - Reduce complexity: 3 EMAs instead of 4, simpler RSI logic
    - Better volatility filtering: avoid choppy sideways markets
    - Cleaner trend alignment: all EMAs must align in same direction
    - Volume confirmation: simple ratio instead of percentile ranking
    - Less signal smoothing: reduce lag from exponential decay
    
    Key changes from trend_momentum_v2:
    - Remove major EMA (200) - too slow for 1h timeframe
    - Simplify RSI to threshold-based (not zone scoring)
    - Use volume ratio instead of percentile (more responsive)
    - Reduce smoothing factor (0.5 vs 0.7) for faster response
    - More conservative leverage (1.5 vs 2.0)
    - Add ADX-like trend strength filter

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

name = "trend_momentum_v3_simplified"
timeframe = "1h"
leverage = 1.5  # More conservative for better risk-adjusted returns

# EMA periods for trend detection (simplified from 4 to 3)
EMA_FAST = 12
EMA_MEDIUM = 26
EMA_SLOW = 50

# RSI configuration (simplified threshold-based)
RSI_PERIOD = 14
RSI_LONG_MIN = 45  # Minimum RSI for long entries
RSI_SHORT_MAX = 55  # Maximum RSI for short entries

# Volume configuration (simple ratio instead of percentile)
VOLUME_LOOKBACK = 20
VOLUME_RATIO_THRESHOLD = 1.2  # Volume must be 20% above average

# Trend strength configuration
TREND_STRENGTH_MIN = 0.002  # Minimum EMA separation ratio

# Volatility configuration
ATR_PERIOD = 14
VOLATILITY_MIN = 0.003  # Avoid too low volatility (choppy)
VOLATILITY_MAX = 0.025  # Avoid too high volatility (risky)
VOLATILITY_TARGET = 0.008  # Target hourly volatility

# Signal configuration
MIN_SIGNAL = 0.15
MAX_SIGNAL = 0.70
SMOOTHING_FACTOR = 0.5  # Less smoothing for faster response


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


def calculate_volume_ratio(volume: np.ndarray, lookback: int = 20) -> np.ndarray:
    """
    Calculate volume ratio vs rolling average.
    Only uses past volume data (no look-ahead).
    
    Args:
        volume: Array of volume values
        lookback: Rolling window for average calculation
    
    Returns:
        Array of volume ratios (current / average)
    """
    n = len(volume)
    volume_ratio = np.ones(n, dtype=np.float64)
    
    if n < lookback:
        return volume_ratio
    
    volume_series = pd.Series(volume)
    
    for i in range(lookback, n):
        window = volume_series.iloc[i-lookback:i]
        avg_volume = window.mean()
        if avg_volume > 0:
            volume_ratio[i] = volume[i] / avg_volume
        else:
            volume_ratio[i] = 1.0
    
    return volume_ratio


def calculate_trend_strength(ema_fast: float, ema_medium: float, 
                             ema_slow: float, close: float) -> float:
    """
    Calculate trend strength based on EMA separation.
    Returns positive for uptrend, negative for downtrend.
    
    Args:
        ema_fast: Fast EMA value
        ema_medium: Medium EMA value
        ema_slow: Slow EMA value
        close: Current close price
    
    Returns:
        Trend strength score (signed)
    """
    if close <= 0 or ema_slow <= 0:
        return 0.0
    
    # Calculate normalized separations
    fast_medium_sep = (ema_fast - ema_medium) / close
    medium_slow_sep = (ema_medium - ema_slow) / close
    
    # Both separations should have same sign for strong trend
    if fast_medium_sep * medium_slow_sep < 0:
        return 0.0  # Conflicting signals
    
    # Combine separations
    trend_strength = (fast_medium_sep + medium_slow_sep) / 2.0
    
    return trend_strength


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Trend Momentum V3 Simplified Strategy.
    
    Signal Logic:
    1. Clean EMA stack alignment (fast > medium > slow for long)
    2. RSI threshold confirmation (not overbought/oversold extremes)
    3. Volume ratio confirmation (above average volume)
    4. Volatility regime filter (avoid choppy/extreme volatility)
    5. Trend strength measurement for signal magnitude
    
    Entry Conditions:
    - LONG: EMA_fast > EMA_medium > EMA_slow + RSI > 45 + volume confirmed
    - SHORT: EMA_fast < EMA_medium < EMA_slow + RSI < 55 + volume confirmed
    
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
    volume_ratio = calculate_volume_ratio(volume, VOLUME_LOOKBACK)
    
    # Determine minimum valid index
    min_valid_index = max(
        EMA_SLOW,
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
        
        # Check volatility regime (avoid choppy or extreme volatility)
        atr_pct = atr[i] / close[i]
        if atr_pct < VOLATILITY_MIN or atr_pct > VOLATILITY_MAX:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Check EMA alignment for trend direction
        ema_aligned_long = (ema_fast[i] > ema_medium[i] > ema_slow[i])
        ema_aligned_short = (ema_fast[i] < ema_medium[i] < ema_slow[i])
        
        if not ema_aligned_long and not ema_aligned_short:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Calculate trend strength
        trend_strength = calculate_trend_strength(
            ema_fast[i], ema_medium[i], ema_slow[i], close[i]
        )
        
        # Filter weak trends
        if abs(trend_strength) < TREND_STRENGTH_MIN:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # RSI confirmation
        rsi_confirmed = False
        if trend_strength > 0:
            # Long: RSI should not be too low (weak momentum)
            rsi_confirmed = rsi[i] >= RSI_LONG_MIN
        else:
            # Short: RSI should not be too high (strong upward momentum)
            rsi_confirmed = rsi[i] <= RSI_SHORT_MAX
        
        if not rsi_confirmed:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume_ratio[i] >= VOLUME_RATIO_THRESHOLD
        
        # Determine signal direction and base magnitude
        if trend_strength > 0:
            # LONG signal
            base_signal = trend_strength * 100.0  # Scale to reasonable range
            if not volume_confirmed:
                base_signal *= 0.7  # Reduce confidence without volume
        else:
            # SHORT signal
            base_signal = trend_strength * 100.0  # Already negative
            if not volume_confirmed:
                base_signal *= 0.7  # Reduce confidence without volume
        
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