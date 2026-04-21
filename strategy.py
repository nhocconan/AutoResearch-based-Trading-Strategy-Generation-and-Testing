#!/usr/bin/env python3
"""
6h_Fibonacci_Extension_Retracement_v1
Hypothesis: Use daily Fibonacci extensions (from prior weekly swing) as dynamic support/resistance.
Long when price retraces to 0.618-0.786 extension during uptrend (price > weekly EMA50).
Short when price retraces to 0.618-0.786 extension during downtrend (price < weekly EMA50).
Uses 60-period volume confirmation to avoid false signals. Designed for 6h timeframe to capture
multi-day moves while avoiding excessive trading. Works in bull/bear via trend filter.
Target: 20-40 trades/year with tight Fibonacci level + volume + trend confluence.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_fibonacci_levels(high, low):
    """Calculate Fibonacci extension levels from swing high/low"""
    diff = high - low
    levels = {
        '0.0': low,
        '0.236': low + 0.236 * diff,
        '0.382': low + 0.382 * diff,
        '0.5': low + 0.5 * diff,
        '0.618': low + 0.618 * diff,
        '0.786': low + 0.786 * diff,
        '1.0': high,
        '1.236': high + 0.236 * diff,
        '1.382': high + 0.382 * diff,
        '1.5': high + 0.5 * diff,
        '1.618': high + 0.618 * diff,
        '2.0': high + 1.0 * diff
    }
    return levels

def calculate_ema(arr, period):
    """Calculate EMA with proper handling"""
    if len(arr) < period:
        return np.full_like(arr, np.nan)
    s = pd.Series(arr)
    return s.ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data once for swing points and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Load weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate daily swing points (look for significant swings)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Find recent significant swing high/low (using 10-day lookback for significance)
    swing_high = np.zeros(len(high_1d))
    swing_low = np.zeros(len(high_1d))
    
    for i in range(10, len(high_1d) - 10):
        # Significant swing high: higher than 10 bars on each side
        if high_1d[i] == np.max(high_1d[i-10:i+11]):
            swing_high[i] = high_1d[i]
        # Significant swing low: lower than 10 bars on each side
        if low_1d[i] == np.min(low_1d[i-10:i+11]):
            swing_low[i] = low_1d[i]
    
    # Get most recent completed swing points
    last_swing_high = 0
    last_swing_low = 0
    for i in range(len(swing_high)-1, 9, -1):
        if swing_high[i] > 0:
            last_swing_high = swing_high[i]
            break
    for i in range(len(swing_low)-1, 9, -1):
        if swing_low[i] > 0:
            last_swing_low = swing_low[i]
            break
    
    # Calculate Fibonacci levels from most recent swing
    if last_swing_high > last_swing_low and last_swing_low > 0:
        fib_levels = calculate_fibonacci_levels(last_swing_high, last_swing_low)
        fib_618 = fib_levels['0.618']
        fib_786 = fib_levels['0.786']
    else:
        # Fallback: use recent high/low
        recent_high = np.max(high_1d[-20:]) if len(high_1d) >= 20 else high_1d[-1]
        recent_low = np.min(low_1d[-20:]) if len(low_1d) >= 20 else low_1d[-1]
        fib_levels = calculate_fibonacci_levels(recent_high, recent_low)
        fib_618 = fib_levels['0.618']
        fib_786 = fib_levels['0.786']
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = calculate_ema(close_1w, 50)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 60-period volume average for confirmation
    volume = prices['volume'].values
    vol_ma_60 = pd.Series(volume).rolling(window=60, min_periods=60).mean().values
    
    signals = np.zeros(n)
    
    for i in range(60, n):
        # Skip if indicators not ready
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_60[i]):
            continue
        
        price = prices['close'].iloc[i]
        vol_current = volume[i]
        vol_average = vol_ma_60[i]
        
        # Volume confirmation: current volume > 1.3 * 60-period average
        volume_ok = vol_current > 1.3 * vol_average if vol_average > 0 else False
        
        # Trend filter: price vs weekly EMA50
        uptrend = price > ema_50_1w_aligned[i]
        downtrend = price < ema_50_1w_aligned[i]
        
        # Fibonacci retracement zone: between 0.618 and 0.786 levels
        in_fib_zone = (price >= fib_618 * 0.995 and price <= fib_786 * 1.005) or \
                      (price >= fib_618 * 0.995 and price <= fib_786 * 1.005)
        
        # Alternative: check if price is near either level (within 0.5%)
        near_618 = abs(price - fib_618) / fib_618 < 0.005
        near_786 = abs(price - fib_786) / fib_786 < 0.005
        near_fib_level = near_618 or near_786
        
        # Entry conditions
        if volume_ok:
            # Long: price near Fib level during uptrend
            if near_fib_level and uptrend:
                signals[i] = 0.25
            # Short: price near Fib level during downtrend
            elif near_fib_level and downtrend:
                signals[i] = -0.25
    
    return signals

name = "6h_Fibonacci_Extension_Retracement_v1"
timeframe = "6h"
leverage = 1.0