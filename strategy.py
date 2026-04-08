#!/usr/bin/env python3
"""
6H Donchian Breakout + Weekly Pivot + Volume Confirmation
Hypothesis: Donchian(20) breakouts from weekly timeframe capture strong momentum,
while 12h EMA(50) provides trend filter. Volume confirmation reduces false breakouts.
Works in bull markets (breakout continuation) and bear markets (breakdown continuation).
Target: 15-35 trades/year per symbol (60-140 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_breakout_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for Donchian channels (20-week high/low)
    df_1w = get_htf_data(prices, '1w')
    
    # Donchian channels (20-week high/low) from previous week
    donchian_high = df_1w['high'].rolling(window=20, min_periods=20).max().shift(1)
    donchian_low = df_1w['low'].rolling(window=20, min_periods=20).min().shift(1)
    
    # 12h EMA(50) for trend filter
    df_12h = get_htf_data(prices, '12h')
    ema_50 = df_12h['close'].ewm(span=50, adjust=False, min_periods=50).mean()
    
    # Align to 6h timeframe
    donchian_high_6h = align_htf_to_ltf(prices, df_1w, donchian_high.values)
    donchian_low_6h = align_htf_to_ltf(prices, df_1w, donchian_low.values)
    ema_50_6h = align_htf_to_ltf(prices, df_12h, ema_50.values)
    
    # Volume filter (>1.3x 50-period average on 6h)
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    vol_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_6h[i]) or np.isnan(donchian_low_6h[i]) or 
            np.isnan(ema_50_6h[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low or trend reverses
            if close[i] <= donchian_low_6h[i] or close[i] < ema_50_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high or trend reverses
            if close[i] >= donchian_high_6h[i] or close[i] > ema_50_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout long at Donchian high with trend alignment
            if (close[i] >= donchian_high_6h[i] and 
                close[i] > ema_50_6h[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Breakout short at Donchian low with trend alignment
            elif (close[i] <= donchian_low_6h[i] and 
                  close[i] < ema_50_6h[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals