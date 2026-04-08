#!/usr/bin/env python3
"""
12h Donchian Breakout with Weekly Trend and Volume Confirmation
Hypothesis: Donchian(20) breakouts from weekly timeframe, filtered by weekly EMA trend and volume spikes,
capture strong momentum moves. Works in both bull and bear markets by following the trend.
Target: 20-40 trades per year per symbol.
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
    
    # Weekly data for Donchian channels and EMA
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Donchian channels (20-period high/low) from weekly data
    high_20 = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min().values
    
    # Weekly EMA(21) for trend filter
    ema_21 = df_1w['close'].ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume filter (>1.5x 20-period average on 12h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    # Align weekly data to 12h timeframe
    high_20_12h = align_htf_to_ltf(prices, df_1w, high_20)
    low_20_12h = align_htf_to_ltf(prices, df_1w, low_20)
    ema_21_12h = align_htf_to_ltf(prices, df_1w, ema_21)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(high_20_12h[i]) or np.isnan(low_20_12h[i]) or 
            np.isnan(ema_21_12h[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below weekly Donchian low or trend reverses
            if close[i] < low_20_12h[i] or close[i] < ema_21_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above weekly Donchian high or trend reverses
            if close[i] > high_20_12h[i] or close[i] > ema_21_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout long at weekly Donchian high with trend alignment
            if (close[i] >= high_20_12h[i] and 
                close[i] > ema_21_12h[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Breakout short at weekly Donchian low with trend alignment
            elif (close[i] <= low_20_12h[i] and 
                  close[i] < ema_21_12h[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals