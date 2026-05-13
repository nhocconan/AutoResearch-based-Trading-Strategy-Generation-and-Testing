#!/usr/bin/env python3
# 6h_Supertrend_FibBreak_1dTrend
# Hypothesis: Combine Supertrend trend detection with Fibonacci retracement breakouts in the direction of the 1d EMA trend.
# Long when price breaks above 61.8% Fib retracement of prior swing low/high AND Supertrend is bullish AND 1d EMA50 is rising.
# Short when price breaks below 38.2% Fib retracement AND Supertrend is bearish AND 1d EMA50 is falling.
# Uses volume confirmation (>1.5x 20-period average) to filter false breakouts.
# Designed for 6H timeframe to capture medium-term moves with low trade frequency (target: 50-150 trades over 4 years).
# Works in bull markets via breakout continuation and in bear markets via retracement-fade logic.

name = "6h_Supertrend_FibBreak_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator."""
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value has no previous close
    
    # Average True Range
    atr = pd.Series(tr).ewm(span=period, adjust=False).mean().values
    
    # Basic Upper and Lower Bands
    hl2 = (high + low) / 2.0
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    # Initialize Supertrend
    supertrend_val = np.zeros_like(close)
    direction = np.ones_like(close)  # 1 for uptrend, -1 for downtrend
    
    # Set first value
    supertrend_val[0] = lower_band[0]
    direction[0] = 1
    
    for i in range(1, len(close)):
        # Update bands
        if upper_band[i] < supertrend_val[i-1] or close[i-1] > supertrend_val[i-1]:
            upper_band[i] = upper_band[i]
        else:
            upper_band[i] = supertrend_val[i-1]
            
        if lower_band[i] > supertrend_val[i-1] or close[i-1] < supertrend_val[i-1]:
            lower_band[i] = lower_band[i]
        else:
            lower_band[i] = supertrend_val[i-1]
        
        # Determine trend
        if close[i] > upper_band[i]:
            direction[i] = 1
        elif close[i] < lower_band[i]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
        
        # Set Supertrend value
        if direction[i] == 1:
            supertrend_val[i] = lower_band[i]
        else:
            supertrend_val[i] = upper_band[i]
    
    return supertrend_val, direction

def fibonacci_levels(high_series, low_series, lookback=20):
    """Calculate Fibonacci retracement levels based on recent swing high/low."""
    highest_high = np.max(high_series[-lookback:])
    lowest_low = np.min(low_series[-lookback:])
    diff = highest_high - lowest_low
    
    # Fibonacci levels
    fib_382 = highest_high - 0.382 * diff
    fib_618 = highest_high - 0.618 * diff
    
    return fib_382, fib_618

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d EMA50 slope (rising/falling)
    ema_slope = np.diff(ema_50_1d, prepend=ema_50_1d[0])
    ema_rising = ema_slope > 0
    ema_falling = ema_slope < 0
    ema_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_rising)
    ema_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_falling)

    # Calculate Supertrend
    st_val, st_dir = supertrend(high, low, close, period=10, multiplier=3.0)

    # Calculate Fibonacci levels (using 20-bar lookback)
    fib_382 = np.full(n, np.nan)
    fib_618 = np.full(n, np.nan)
    
    for i in range(20, n):
        fib_382[i], fib_618[i] = fibonacci_levels(high[:i+1], low[:i+1], lookback=20)

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(st_val[i]) or 
            np.isnan(fib_382[i]) or np.isnan(fib_618[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above 61.8% Fib + Supertrend bullish + 1d EMA50 rising + volume spike
            if (close[i] > fib_618[i] and 
                st_val[i] < close[i] and  # Supertrend is below price (bullish)
                ema_rising_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below 38.2% Fib + Supertrend bearish + 1d EMA50 falling + volume spike
            elif (close[i] < fib_382[i] and 
                  st_val[i] > close[i] and  # Supertrend is above price (bearish)
                  ema_falling_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Supertrend OR 1d EMA50 turns flat/falling
            if (close[i] < st_val[i] or not ema_rising_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Supertrend OR 1d EMA50 turns flat/rising
            if (close[i] > st_val[i] or not ema_falling_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals