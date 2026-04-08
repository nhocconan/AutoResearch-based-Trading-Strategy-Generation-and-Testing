#!/usr/bin/env python3
"""
12h Donchian Breakout + Weekly Trend + Volume Confirmation v1
Hypothesis: Price breaking above/below 20-period Donchian channels on 12h timeframe
with weekly trend alignment (EMA50) and volume confirmation captures strong trends
while avoiding whipsaws. Designed for 12h to limit trade frequency and reduce fee drag.
Target: 12-37 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_weekly_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA(50) for trend filter
    ema_50 = df_1w['close'].ewm(span=50, adjust=False, min_periods=50).mean()
    
    # 12h Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter (>1.8x 30-period average on 12h)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_filter = volume > (vol_ma * 1.8)
    
    # Align weekly EMA to 12h timeframe
    ema_50_12h = align_htf_to_ltf(prices, df_1w, ema_50.values)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_50_12h[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low or trend reverses
            if close[i] <= low_20[i] or close[i] < ema_50_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high or trend reverses
            if close[i] >= high_20[i] or close[i] > ema_50_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long breakout above Donchian high with trend alignment
            if (close[i] >= high_20[i] and 
                close[i] > ema_50_12h[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short breakdown below Donchian low with trend alignment
            elif (close[i] <= low_20[i] and 
                  close[i] < ema_50_12h[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals