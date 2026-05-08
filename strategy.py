#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1w pivot direction and volume confirmation
# Long when price breaks above 20-period high AND weekly pivot shows bullish bias (close > weekly pivot)
# Short when price breaks below 20-period low AND weekly pivot shows bearish bias (close < weekly pivot)
# Volume confirmation: current volume > 1.5 * 20-period average
# Weekly pivot calculated as (weekly high + weekly low + weekly close) / 3
# Designed for low trade frequency with strong trend continuation signals
# Target: 50-150 total trades over 4 years = 12-37/year

name = "6h_Donchian20_1wPivot_Volume"
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
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot: (high + low + close) / 3
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Donchian channels on 6h data
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        weekly_pivot_val = weekly_pivot_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Enter long: price breaks above 20-period high AND close > weekly pivot AND volume confirmation
            if (close[i] > high_20[i] and 
                close[i] > weekly_pivot_val and 
                vol_conf):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below 20-period low AND close < weekly pivot AND volume confirmation
            elif (close[i] < low_20[i] and 
                  close[i] < weekly_pivot_val and 
                  vol_conf):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below 20-period low OR weekly pivot turns bearish
            if (close[i] < low_20[i] or 
                close[i] < weekly_pivot_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above 20-period high OR weekly pivot turns bullish
            if (close[i] > high_20[i] or 
                close[i] > weekly_pivot_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals