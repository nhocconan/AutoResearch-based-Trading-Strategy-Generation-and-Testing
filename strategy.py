#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h_1w_1d_elliott_wave_v1
# Uses weekly Elliott Wave structure to identify impulse vs corrective phases,
# then trades pullbacks to the 38.2% Fibonacci retracement within the trend.
# In bull markets: buy pullbacks in uptrends; in bear markets: sell rallies in downtrends.
# Uses 1d ADX > 25 to confirm trend strength and 6h volume > 1.5x average for confirmation.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "6h_1w_1d_elliott_wave_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Elliott Wave structure
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Identify weekly swing points (simplified Elliott Wave)
    # Look for local maxima/minima over 3-week window
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Find weekly pivot highs and lows
    def find_pivots(arr, window=3):
        pivots = np.full_like(arr, np.nan)
        for i in range(window, len(arr) - window):
            if arr[i] == np.max(arr[i-window:i+window+1]):
                pivots[i] = arr[i]  # pivot high
            elif arr[i] == np.min(arr[i-window:i+window+1]):
                pivots[i] = arr[i]  # pivot low
        return pivots
    
    pivot_highs = find_pivots(high_1w, 3)
    pivot_lows = find_pivots(low_1w, 3)
    
    # Get most recent completed pivot points
    last_pivot_high = np.nan
    last_pivot_low = np.nan
    for i in range(len(pivot_highs)-1, -1, -1):
        if not np.isnan(pivot_highs[i]):
            last_pivot_high = pivot_highs[i]
            break
    for i in range(len(pivot_lows)-1, -1, -1):
        if not np.isnan(pivot_lows[i]):
            last_pivot_low = pivot_lows[i]
            break
    
    # Calculate Fibonacci levels (38.2% retracement)
    if not np.isnan(last_pivot_high) and not np.isnan(last_pivot_low):
        wave_range = last_pivot_high - last_pivot_low
        fib_382 = last_pivot_low + wave_range * 0.382
        fib_618 = last_pivot_low + wave_range * 0.618
    else:
        fib_382 = np.nan
        fib_618 = np.nan
    
    # Align weekly Fibonacci levels to 6h timeframe
    fib_382_level = align_htf_to_ltf(prices, df_1w, np.full_like(close_1w, fib_382))
    fib_618_level = align_htf_to_ltf(prices, df_1w, np.full_like(close_1w, fib_618))
    
    # Get daily data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate ADX on daily timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    
    # Wilder's smoothing
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return result
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_1d = wilders_smooth(tr, 14)
    plus_dm_smooth = wilders_smooth(plus_dm, 14)
    minus_dm_smooth = wilders_smooth(minus_dm, 14)
    
    plus_di = np.where(atr_1d != 0, 100 * plus_dm_smooth / atr_1d, 0)
    minus_di = np.where(atr_1d != 0, 100 * minus_dm_smooth / atr_1d, 0)
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx_1d = wilders_smooth(dx, 14)
    adx_filter = align_htf_to_ltf(prices, df_1d, adx_1d) > 25  # strong trend
    
    # Volume confirmation on 6h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if levels not ready
        if (np.isnan(fib_382_level[i]) or np.isnan(fib_618_level[i]) or 
            np.isnan(adx_filter[i])):
            signals[i] = 0.0
            continue
        
        # Require both trend and volume filters
        if not (adx_filter[i] and vol_confirm[i]):
            # Hold current position if filters fail
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine trend direction from weekly structure
        # Uptrend: last pivot high > previous pivot high
        # Downtrend: last pivot low < previous pivot low
        # Simplified: if price > fib_618, consider uptrend; else downtrend
        if close[i] > fib_618_level[i]:
            # Uptrend mode: buy pullbacks to 38.2% fib level
            if close[i] <= fib_382_level[i] * 1.005 and position != 1:  # allow 0.5% slippage
                position = 1
                signals[i] = 0.25
            elif close[i] > fib_618_level[i] and position == 1:
                position = 0
                signals[i] = 0.0
        else:
            # Downtrend mode: sell rallies to 38.2% fib level
            if close[i] >= fib_382_level[i] * 0.995 and position != -1:  # allow 0.5% slippage
                position = -1
                signals[i] = -0.25
            elif close[i] < fib_382_level[i] and position == -1:
                position = 0
                signals[i] = 0.0
        
        # Hold position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals