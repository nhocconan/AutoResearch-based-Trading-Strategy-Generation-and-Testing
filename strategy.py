#!/usr/bin/env python3
"""
strategy.py - SMI Momentum Trend Hybrid V1
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

Strategy Hypothesis:
    Stochastic Momentum Index (SMI) with trend filter on 4h timeframe.
    
    Why this works:
    - SMI is smoother than RSI/Stochastic, less whipsaw noise
    - 4h timeframe captures sustained momentum moves in crypto
    - EMA-200 filter ensures we trade with major trend direction
    - SMI extremes (±40) identify momentum exhaustion points
    - Volatility-based signal scaling controls position size during high vol
    
    Key differences from failed strategies:
    - Not pure mean reversion (failed with BB/KC squeeze)
    - Not pure trend following (Supertrend had -84% DD)
    - Momentum entry WITH trend filter = best of both worlds
    - 4h should reduce trade frequency but improve quality vs 1h
    
    Risk Management:
    - Signal magnitude scales with volatility (lower sig in high vol)
    - SMI threshold ensures we only trade at momentum extremes
    - Trend filter prevents counter-trend trades (major DD source)

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

name = "smi_momentum_trend_4h_v1"
timeframe = "4h"
leverage = 1.5  # Conservative leverage for better risk-adjusted returns

# SMI Configuration (Blau's original parameters)
SMI_K = 5      # First smoothing factor
SMI_R = 34     # Lookback period for stochastic
SMI_S = 3      # Second smoothing factor
SMI_LONG_THRESHOLD = -40   # SMI below this = oversold (long signal)
SMI_SHORT_THRESHOLD = 40   # SMI above this = overbought (short signal)
SMI_NEUTRAL = 15           # Hysteresis band around zero

# Trend Filter Configuration
EMA_TREND = 200            # Major trend filter
EMA_FAST = 50              # Secondary trend confirmation
TREND_STRENGTH_MIN = 0.02  # Minimum trend strength to trade

# Volatility Configuration
ATR_PERIOD = 14
VOLATILITY_TARGET = 0.020  # Target ATR as % of price
VOLATILITY_MIN = 0.005     # Minimum ATR % to trade (avoid dead markets)
VOLATILITY_MAX = 0.080     # Maximum ATR % to trade (avoid chaos)

# Signal Configuration
MIN_SIGNAL_MAGNITUDE = 0.20  # Minimum signal to generate position
MAX_SIGNAL = 0.90            # Maximum signal magnitude
SMOOTHING_FACTOR = 0.40      # EMA smoothing for signals
HYSTERESIS_THRESHOLD = 0.15  # Minimum change to flip signal direction

# Volume Confirmation
VOLUME_LOOKBACK = 20
VOLUME_MIN_RATIO = 0.50      # Volume must be at least this % of average


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


def calculate_smi(close: np.ndarray, 
                  high: np.ndarray, 
                  low: np.ndarray,
                  k: int = 5,
                  r: int = 34,
                  s: int = 3) -> np.ndarray:
    """
    Calculate Stochastic Momentum Index (Blau, 1994).
    
    SMI = 100 * EMA(EMA(C - midpoint, k), s) / EMA(EMA(HL_range, k), s)
    
    Where midpoint = EMA(EMA((H+L)/2, r), r)
    
    This is a double-smoothed momentum oscillator ranging roughly -100 to +100.
    Only uses past data (no look-ahead).
    """
    n = len(close)
    smi = np.zeros(n, dtype=np.float64)
    
    if n < r + k + s + 5:
        return smi
    
    # Calculate midpoint (double-smoothed center of HL range)
    hl_mid = (high + low) / 2.0
    hl_mid_series = pd.Series(hl_mid)
    midpoint = hl_mid_series.ewm(span=r, adjust=False, min_periods=r).mean()
    midpoint = midpoint.ewm(span=r, adjust=False, min_periods=r).mean().values
    midpoint = np.nan_to_num(midpoint, nan=0.0)
    
    # Calculate distance from midpoint
    distance = close - midpoint
    
    # Calculate HL range
    hl_range = high - low
    hl_range_series = pd.Series(hl_range)
    
    # Double-smooth the distance and range
    distance_series = pd.Series(distance)
    
    # First smoothing (k)
    dist_smooth1 = distance_series.ewm(span=k, adjust=False, min_periods=k).mean()
    range_smooth1 = hl_range_series.ewm(span=k, adjust=False, min_periods=k).mean()
    
    # Second smoothing (s)
    dist_smooth2 = dist_smooth1.ewm(span=s, adjust=False, min_periods=s).mean()
    range_smooth2 = range_smooth1.ewm(span=s, adjust=False, min_periods=s).mean()
    
    # Calculate SMI
    smi_values = 100.0 * dist_smooth2.values / range_smooth2.values.replace(0, np.inf)
    smi = np.nan_to_num(smi_values, nan=0.0)
    
    # Clip to reasonable range
    smi = np.clip(smi, -100.0, 100.0)
    
    return smi


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


def calculate_trend_strength(close: np.ndarray, 
                             ema_trend: np.ndarray, 
                             ema_fast: np.ndarray) -> np.ndarray:
    """
    Calculate trend strength as normalized distance from trend EMA.
    Returns value in [0, 1] where higher = stronger trend.
    """
    n = len(close)
    strength = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        if ema_trend[i] <= 0:
            strength[i] = 0.0
            continue
        
        # Distance from major trend EMA
        dist_from_trend = (close[i] - ema_trend[i]) / ema_trend[i]
        
        # Fast EMA confirmation
        ema_alignment = np.sign(close[i] - ema_trend[i]) * np.sign(ema_fast[i] - ema_trend[i])
        
        # Combine: distance + alignment
        strength[i] = abs(dist_from_trend) * 100 * (1.0 if ema_alignment > 0 else 0.5)
    
    return np.clip(strength, 0.0, 1.0)


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    SMI Momentum Trend Hybrid V1 Strategy.
    
    Signal Logic:
    1. Calculate SMI momentum oscillator (double-smoothed stochastic)
    2. Filter by major trend direction (price vs EMA-200)
    3. Only trade SMI extremes in direction of trend
    4. Scale signal by volatility (reduce size in high vol)
    5. Smooth signals with EMA
    6. Apply hysteresis to reduce whipsaws
    7. Filter by minimum signal magnitude
    
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
    ema_trend = calculate_ema(close, EMA_TREND)
    ema_fast = calculate_ema(close, EMA_FAST)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    volume_ratio = calculate_volume_ratio(volume, VOLUME_LOOKBACK)
    smi = calculate_smi(close, high, low, SMI_K, SMI_R, SMI_S)
    trend_strength = calculate_trend_strength(close, ema_trend, ema_fast)
    
    # Calculate minimum valid index (all indicators need warmup)
    min_valid_index = max(
        EMA_TREND,
        EMA_FAST,
        ATR_PERIOD + 1,
        VOLUME_LOOKBACK,
        SMI_R + SMI_K + SMI_S + 10
    )
    
    # Generate signals
    prev_signal = 0.0
    prev_direction = 0
    
    for i in range(min_valid_index, n):
        # Skip invalid bars
        if close[i] <= 0 or atr[i] <= 0 or ema_trend[i] <= 0:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # Volatility filter (not too low, not too high)
        atr_pct = atr[i] / close[i]
        if atr_pct < VOLATILITY_MIN or atr_pct > VOLATILITY_MAX:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # Volume filter (ensure sufficient liquidity)
        if volume_ratio[i] < VOLUME_MIN_RATIO:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # Trend filter - only trade with major trend
        trend_direction = np.sign(close[i] - ema_trend[i])
        trend_str = trend_strength[i]
        
        if trend_str < TREND_STRENGTH_MIN:
            # Trend too weak, stay neutral
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # SMI momentum signal
        smi_value = smi[i]
        raw_signal = 0.0
        
        if trend_direction > 0:
            # Uptrend: look for long entries on SMI oversold (pullback)
            if smi_value < SMI_LONG_THRESHOLD:
                # Strong oversold in uptrend = buy signal
                raw_signal = 1.0 * (abs(smi_value) / 100.0)
            elif smi_value > SMI_SHORT_THRESHOLD:
                # Overbought in uptrend = reduce/exit (but don't short)
                raw_signal = 0.0
        elif trend_direction < 0:
            # Downtrend: look for short entries on SMI overbought (rally)
            if smi_value > SMI_SHORT_THRESHOLD:
                # Strong overbought in downtrend = sell signal
                raw_signal = -1.0 * (abs(smi_value) / 100.0)
            elif smi_value < SMI_LONG_THRESHOLD:
                # Oversold in downtrend = reduce/exit (but don't long)
                raw_signal = 0.0
        
        # Apply trend strength multiplier
        raw_signal *= trend_str
        
        # Volatility normalization (scale by target volatility)
        vol_factor = VOLATILITY_TARGET / max(atr_pct, 0.001)
        vol_factor = np.clip(vol_factor, 0.5, 1.5)
        raw_signal *= vol_factor
        
        # Signal smoothing (EMA on signals)
        smoothed_signal = SMOOTHING_FACTOR * prev_signal + (1.0 - SMOOTHING_FACTOR) * raw_signal
        
        # Hysteresis: don't flip direction on small changes
        current_direction = np.sign(smoothed_signal)
        if current_direction != 0 and current_direction != prev_direction:
            if abs(smoothed_signal - prev_signal) < HYSTERESIS_THRESHOLD:
                smoothed_signal = prev_signal
        
        # Apply minimum magnitude filter
        if abs(smoothed_signal) < MIN_SIGNAL_MAGNITUDE:
            smoothed_signal = 0.0
        
        # Clip to max signal
        signal = np.clip(smoothed_signal, -MAX_SIGNAL, MAX_SIGNAL)
        
        signals[i] = signal
        prev_signal = signal
        prev_direction = np.sign(signal)
    
    return signals