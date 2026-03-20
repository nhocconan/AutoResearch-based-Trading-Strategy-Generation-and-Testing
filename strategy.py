#!/usr/bin/env python3
"""
strategy.py - Adaptive Trend Follower V3
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "1h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Learning from #027 failure (Sharpe=-0.064), simplifying while keeping
    what worked in #002 (Sharpe=0.330). Key changes:
    
    - Cleaner trend detection with ADX for trend strength filtering
    - Simplified EMA stack (3 EMAs instead of 4)
    - Better volatility-based position sizing (ATR normalized)
    - Reduced signal smoothing to minimize lag
    - Add mean-reversion component for range-bound markets
    - Remove volume percentile (less reliable on 1h)
    
    Core Logic:
    1. ADX > 25 → trend-following mode (EMA alignment)
    2. ADX < 20 → mean-reversion mode (RSI extremes)
    3. ADX 20-25 → reduced position size
    4. Volatility scaling inversely proportional to ATR
    5. Simple signal smoothing (less aggressive than before)

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

name = "adaptive_trend_v3"
timeframe = "1h"
leverage = 2.0  # Conservative leverage for risk management

# EMA periods for trend detection
EMA_FAST = 12
EMA_MEDIUM = 26
EMA_SLOW = 50

# ADX configuration for trend strength
ADX_PERIOD = 14
ADX_TREND_THRESHOLD = 25  # ADX above this = strong trend
ADX_RANGE_THRESHOLD = 20  # ADX below this = range-bound

# RSI configuration
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
RSI_EXTREME_HIGH = 80
RSI_EXTREME_LOW = 20

# Volatility configuration
ATR_PERIOD = 14
VOLATILITY_TARGET = 0.012  # Target hourly volatility
VOLATILITY_MIN = 0.003
VOLATILITY_MAX = 0.040

# Signal configuration
MIN_SIGNAL = 0.15
MAX_SIGNAL = 0.75
SMOOTHING_FACTOR = 0.5  # Less smoothing for responsiveness

# Position sizing
TREND_MODE_SIZE = 1.0
RANGE_MODE_SIZE = 0.6
TRANSITION_MODE_SIZE = 0.8


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


def calculate_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calculate Average Directional Index using only past data.
    
    Args:
        high: Array of high prices
        low: Array of low prices
        close: Array of close prices
        period: ADX period
    
    Returns:
        Array of ADX values (0-100)
    """
    n = len(close)
    adx = np.zeros(n, dtype=np.float64)
    
    if n < period * 2 + 1:
        return adx
    
    # Calculate True Range and Directional Movement
    tr = np.zeros(n, dtype=np.float64)
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(high[i] - high[i-1], 0)
        else:
            plus_dm[i] = 0
            
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(low[i-1] - low[i], 0)
        else:
            minus_dm[i] = 0
    
    # Smooth TR, +DM, -DM
    tr_series = pd.Series(tr)
    plus_dm_series = pd.Series(plus_dm)
    minus_dm_series = pd.Series(minus_dm)
    
    atr_smooth = tr_series.ewm(span=period, adjust=False, min_periods=period).mean().values
    plus_dm_smooth = plus_dm_series.ewm(span=period, adjust=False, min_periods=period).mean().values
    minus_dm_smooth = minus_dm_series.ewm(span=period, adjust=False, min_periods=period).mean().values
    
    # Calculate DI+ and DI-
    plus_di = np.zeros(n, dtype=np.float64)
    minus_di = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        if atr_smooth[i] > 0:
            plus_di[i] = 100.0 * plus_dm_smooth[i] / atr_smooth[i]
            minus_di[i] = 100.0 * minus_dm_smooth[i] / atr_smooth[i]
    
    # Calculate DX
    dx = np.zeros(n, dtype=np.float64)
    for i in range(n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    # Calculate ADX (smoothed DX)
    dx_series = pd.Series(dx)
    adx_series = dx_series.ewm(span=period, adjust=False, min_periods=period).mean()
    
    adx = np.nan_to_num(adx_series.values, nan=0.0)
    
    return adx


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Adaptive Trend Follower V3 Strategy.
    
    Signal Logic:
    1. ADX determines market regime (trend vs range)
    2. Trend mode: EMA alignment + RSI confirmation
    3. Range mode: RSI mean-reversion at extremes
    4. Volatility-based position sizing
    5. Light signal smoothing
    
    Entry Conditions:
    - LONG (trend): EMA_fast > EMA_med > EMA_slow + ADX > 25 + RSI < 70
    - SHORT (trend): EMA_fast < EMA_med < EMA_slow + ADX > 25 + RSI > 30
    - LONG (range): RSI < 25 + ADX < 20
    - SHORT (range): RSI > 75 + ADX < 20
    
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
    adx = calculate_adx(high, low, close, ADX_PERIOD)
    
    # Determine minimum valid index
    min_valid_index = max(
        EMA_SLOW,
        RSI_PERIOD + 1,
        ATR_PERIOD + 1,
        ADX_PERIOD * 2 + 1
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
        
        # Determine market regime based on ADX
        if adx[i] >= ADX_TREND_THRESHOLD:
            # TREND MODE
            regime_size = TREND_MODE_SIZE
            
            # Check EMA alignment for trend direction
            bullish_alignment = (ema_fast[i] > ema_medium[i] > ema_slow[i])
            bearish_alignment = (ema_fast[i] < ema_medium[i] < ema_slow[i])
            
            if bullish_alignment:
                # LONG signal in uptrend
                # RSI confirmation (not overbought)
                if rsi[i] < RSI_OVERBOUGHT:
                    rsi_factor = 1.0 - (rsi[i] - 50) / 50  # Scale 1.0 at RSI=50, 0 at RSI=100
                    rsi_factor = max(0.3, rsi_factor)
                    base_signal = 1.0 * rsi_factor
                else:
                    base_signal = 0.0
                    
            elif bearish_alignment:
                # SHORT signal in downtrend
                # RSI confirmation (not oversold)
                if rsi[i] > RSI_OVERSOLD:
                    rsi_factor = (rsi[i] - 50) / 50  # Scale 1.0 at RSI=100, 0 at RSI=50
                    rsi_factor = max(0.3, rsi_factor)
                    base_signal = -1.0 * rsi_factor
                else:
                    base_signal = 0.0
            else:
                # No clear alignment
                base_signal = 0.0
                
        elif adx[i] <= ADX_RANGE_THRESHOLD:
            # RANGE MODE (mean reversion)
            regime_size = RANGE_MODE_SIZE
            
            if rsi[i] <= RSI_EXTREME_LOW:
                # Oversold → LONG
                base_signal = 1.0 * (1.0 - rsi[i] / RSI_EXTREME_LOW)
            elif rsi[i] >= RSI_EXTREME_HIGH:
                # Overbought → SHORT
                base_signal = -1.0 * ((rsi[i] - RSI_EXTREME_HIGH) / (100 - RSI_EXTREME_HIGH))
            else:
                base_signal = 0.0
                
        else:
            # TRANSITION MODE (reduced size)
            regime_size = TRANSITION_MODE_SIZE
            
            # Weaker signals in transition
            if ema_fast[i] > ema_medium[i] and rsi[i] < 60:
                base_signal = 0.5
            elif ema_fast[i] < ema_medium[i] and rsi[i] > 40:
                base_signal = -0.5
            else:
                base_signal = 0.0
        
        # Apply regime size scaling
        raw_signal = base_signal * regime_size
        
        # Volatility-based position sizing (inverse relationship)
        vol_factor = VOLATILITY_TARGET / max(atr_pct, 0.001)
        vol_factor = np.clip(vol_factor, 0.6, 1.8)
        
        raw_signal = raw_signal * vol_factor
        
        # Apply exponential smoothing (lighter than before)
        smoothed_signal = SMOOTHING_FACTOR * prev_signal + (1.0 - SMOOTHING_FACTOR) * raw_signal
        prev_signal = smoothed_signal
        
        # Apply thresholds
        if abs(smoothed_signal) < MIN_SIGNAL:
            smoothed_signal = 0.0
        
        # Clip to valid range
        signal = np.clip(smoothed_signal, -MAX_SIGNAL, MAX_SIGNAL)
        
        signals[i] = signal
    
    return signals