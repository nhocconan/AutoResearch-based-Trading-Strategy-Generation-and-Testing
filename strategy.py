#!/usr/bin/env python3
"""
12h Donchian Breakout with Weekly Trend and Volume Confirmation
Hypothesis: 12h Donchian(20) breakouts aligned with weekly trend (EMA40) and volume > 1.5x average capture strong momentum.
Designed for 12h timeframe to limit trades (target: 12-37/year) and avoid fee drag. Works in bull via breakouts and bear via short breakdowns.
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
    ema_40 = df_1w['close'].ewm(span=40, adjust=False).mean().values
    ema_40_12h = align_htf_to_ltf(prices, df_1w, ema_40)
    
    # 12h Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter (>1.5x 20-period average on 12h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(ema_40_12h[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low or trend reverses
            if close[i] <= low_roll[i] or close[i] < ema_40_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high or trend reverses
            if close[i] >= high_roll[i] or close[i] > ema_40_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout long above Donchian high with trend and volume
            if (close[i] > high_roll[i] and 
                close[i] > ema_40_12h[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Breakdown short below Donchian low with trend and volume
            elif (close[i] < low_roll[i] and 
                  close[i] < ema_40_12h[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals