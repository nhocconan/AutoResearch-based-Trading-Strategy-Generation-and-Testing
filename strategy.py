#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot context and volume confirmation
# Long when price breaks above Donchian(20) high and weekly pivot > prior weekly pivot (uptrend)
# Short when price breaks below Donchian(20) low and weekly pivot < prior weekly pivot (downtrend)
# Weekly pivot provides higher-timeframe trend bias to avoid counter-trend trades
# Volume confirmation ensures breakout authenticity
# Designed for low trade frequency in both bull and bear markets
# Target: 50-150 total trades over 4 years = 12-37/year

name = "6h_Donchian20_WeeklyPivot_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard formula)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Pivot Point = (High + Low + Close) / 3
    pivot = (high_1w + low_1w + close_1w) / 3.0
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    
    # Donchian(20) channels on 6h data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        pivot_val = pivot_aligned[i]
        donch_high = donchian_high[i]
        donch_low = donchian_low[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: break above Donchian high + weekly pivot up + volume spike
            if (close[i] > donch_high and 
                i > 0 and pivot_val > pivot_aligned[i-1] and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: break below Donchian low + weekly pivot down + volume spike
            elif (close[i] < donch_low and 
                  i > 0 and pivot_val < pivot_aligned[i-1] and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: break below Donchian low
            if close[i] < donch_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: break above Donchian high
            if close[i] > donch_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals