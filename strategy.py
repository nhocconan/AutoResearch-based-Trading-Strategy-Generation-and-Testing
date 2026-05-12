#!/usr/bin/env python3
name = "6h_Donchian20_WeeklyPivotDir_20pVol"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Donchian channels (20) ===
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 12h Weekly pivot direction (weekly high/low/close) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Weekly pivot: P = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    r1_12h = 2 * pivot_12h - low_12h
    s1_12h = 2 * pivot_12h - high_12h
    
    # Pivot direction: bullish if close > pivot, bearish if close < pivot
    pivot_dir_12h = np.where(close_12h > pivot_12h, 1, np.where(close_12h < pivot_12h, -1, 0))
    
    pivot_dir_aligned = align_htf_to_ltf(prices, df_12h, pivot_dir_12h.astype(float))
    
    # === 20-period volume average filter ===
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > vol_avg_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(high_max_20[i]) or 
            np.isnan(low_min_20[i]) or
            np.isnan(pivot_dir_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Break above Donchian high + weekly pivot bullish + volume filter
            if (close[i] > high_max_20[i] and
                pivot_dir_aligned[i] > 0 and
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low + weekly pivot bearish + volume filter
            elif (close[i] < low_min_20[i] and
                  pivot_dir_aligned[i] < 0 and
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Close below Donchian low or pivot turns bearish
            if close[i] < low_min_20[i] or pivot_dir_aligned[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Close above Donchian high or pivot turns bullish
            if close[i] > high_max_20[i] or pivot_dir_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals