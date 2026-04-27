#!/usr/bin/env python3
"""
6h Donchian(20) breakout + weekly pivot direction + volume confirmation
Long when price breaks above Donchian upper + weekly pivot up + volume spike
Short when price breaks below Donchian lower + weekly pivot down + volume spike
Exit when price returns to Donchian middle (mean) or weekly pivot flips
Designed for low frequency (12-37/year) to minimize fee drag
Uses weekly pivot for trend filter and Donchian for breakout timing
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation (trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot levels (using prior week's OHLC)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    open_1w = df_1w['open'].values
    
    # Weekly Pivot Point (PP) = (H + L + C) / 3
    pp = (high_1w + low_1w + close_1w) / 3.0
    # Weekly Range = H - L
    range_1w = high_1w - low_1w
    # Support and Resistance levels
    r1 = (2 * pp) - low_1w
    s1 = (2 * pp) - high_1w
    r2 = pp + range_1w
    s2 = pp - range_1w
    
    # Align weekly pivot levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # Donchian channel (20-period) for breakout signals
    lookback = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    middle = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        upper[i] = np.max(high[i-lookback+1:i+1])
        lower[i] = np.min(low[i-lookback+1:i+1])
        middle[i] = (upper[i] + lower[i]) / 2.0
    
    # Volume filter: volume > 1.8x average (to avoid false breakouts)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need weekly pivot + Donchian (20) + volume MA (20)
    start_idx = max(20, 20)  # weekly pivot needs at least 1 week, Donchian needs 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(middle[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        
        # Current indicators
        pp_level = pp_aligned[i]
        upper_level = upper[i]
        lower_level = lower[i]
        middle_level = middle[i]
        
        # Weekly trend: price vs weekly pivot
        weekly_up = price_now > pp_level
        weekly_down = price_now < pp_level
        
        # Volume filter: volume > 1.8x average
        vol_filter = vol_now > 1.8 * vol_ma_20[i]
        
        if position == 0:
            # Bull: price breaks above Donchian upper + weekly trend up + volume spike
            if price_now > upper_level and weekly_up and vol_filter:
                signals[i] = size
                position = 1
            # Bear: price breaks below Donchian lower + weekly trend down + volume spike
            elif price_now < lower_level and weekly_down and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to Donchian middle or weekly trend turns down
            if price_now < middle_level or not weekly_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to Donchian middle or weekly trend turns up
            if price_now > middle_level or not weekly_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian_20_WeeklyPivot_Volume"
timeframe = "6h"
leverage = 1.0