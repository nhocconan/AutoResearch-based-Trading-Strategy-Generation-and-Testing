#!/usr/bin/env python3
"""
6h_HTF_WeeklyPivot_Donchian20_Breakout_V1
Hypothesis: Use weekly pivot (R1/S1) from 1w timeframe as structural bias + 6h Donchian(20) breakout with volume confirmation (>1.5x 20-bar MA). Weekly pivot provides multi-day support/resistance that works in both bull (buy weakness at S1) and bear (sell strength at R1) markets. Donchian breakout captures momentum, volume filter avoids fakeouts. Target 12-25 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')  # for weekly pivot
    df_1d = get_htf_data(prices, '1d')  # for Donchian context (optional trend filter)
    
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # === Weekly Pivot (R1, S1) ===
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    # Weekly pivot point
    pivot_w = (high_w + low_w + close_w) / 3.0
    # Weekly R1 and S1 (standard pivot formula)
    weekly_r1 = 2 * pivot_w - low_w
    weekly_s1 = 2 * pivot_w - high_w
    
    # Align to 6h timeframe
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # === 6h Indicators ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume MA (20-period) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i]) 
            or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) 
            or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ok = vol > 1.5 * vol_ma[i]  # volume confirmation
        
        if position == 0:
            # Long: price breaks above Donchian high AND above weekly S1 (bullish bias)
            if price > highest_high[i-1] and price > weekly_s1_aligned[i-1] and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND below weekly R1 (bearish bias)
            elif price < lowest_low[i-1] and price < weekly_r1_aligned[i-1] and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price breaks below Donchian low OR weekly S1 broken
            if price < lowest_low[i-1] or price < weekly_s1_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price breaks above Donchian high OR weekly R1 broken
            if price > highest_high[i-1] or price > weekly_r1_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_HTF_WeeklyPivot_Donchian20_Breakout_V1"
timeframe = "6h"
leverage = 1.0