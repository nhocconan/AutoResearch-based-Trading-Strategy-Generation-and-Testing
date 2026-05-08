#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WeeklyPivot_Donchian20_Trend_Filter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Get daily data for trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate weekly pivot and weekly range from previous week
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    high_1w_prev = np.roll(high_1w, 1)
    low_1w_prev = np.roll(low_1w, 1)
    close_1w_prev = np.roll(close_1w, 1)
    high_1w_prev[0] = np.nan
    low_1w_prev[0] = np.nan
    close_1w_prev[0] = np.nan
    
    pivot_1w = (high_1w_prev + low_1w_prev + close_1w_prev) / 3
    range_1w = high_1w_prev - low_1w_prev
    r1_1w = pivot_1w + (range_1w * 1.0 / 4)
    s1_1w = pivot_1w - (range_1w * 1.0 / 4)
    
    # Align weekly levels to 6h
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # Daily EMA(50) for trend
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Donchian(20) channels from previous 20 periods
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume filter: volume > 1.5 * 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        r1 = r1_1w_aligned[i]
        s1 = s1_1w_aligned[i]
        ema50 = ema50_1d_aligned[i]
        upper_donchian = highest_20[i]
        lower_donchian = lowest_20[i]
        vol_filt = volume_filter[i]
        
        if position == 0:
            # Long: price breaks above weekly R1 + above Donchian high + uptrend + volume
            if (close[i] > r1 and 
                close[i] > upper_donchian and 
                close[i] > ema50 and 
                vol_filt):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S1 + below Donchian low + downtrend + volume
            elif (close[i] < s1 and 
                  close[i] < lower_donchian and 
                  close[i] < ema50 and 
                  vol_filt):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below weekly S1 OR trend turns down
            if (close[i] < s1 or close[i] < ema50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above weekly R1 OR trend turns up
            if (close[i] > r1 or close[i] > ema50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals