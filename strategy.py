#!/usr/bin/env python3
"""
12h_1w_24h_Volume_Spike_Breakout_v1
Hypothesis: Use weekly high/low as breakout levels with volume spike confirmation on 12h timeframe.
Long when price breaks above weekly high with volume > 2x 24-period average,
short when breaks below weekly low with volume > 2x 24-period average.
Weekly high/low provides strong institutional levels; volume confirms breakout strength.
Designed for low trade frequency (target: 50-150 total over 4 years) to minimize fee drag.
Works in bull via breakouts above weekly high, in bear via breakdowns below weekly low.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_24h_Volume_Spike_Breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for high/low levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Previous week's high and low
    prev_high = df_1w['high'].iloc[-2] if len(df_1w) >= 2 else df_1w['high'].iloc[-1]
    prev_low = df_1w['low'].iloc[-2] if len(df_1w) >= 2 else df_1w['low'].iloc[-1]
    
    # Weekly high and low arrays
    weekly_high_array = np.full(len(df_1w), prev_high)
    weekly_low_array = np.full(len(df_1w), prev_low)
    
    # Align weekly levels to 12h timeframe
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high_array)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low_array)
    
    # Volume confirmation: current volume > 2x 24-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=24, min_periods=24).mean()
    vol_ratio = volume_series / vol_ma
    vol_ratio = vol_ratio.fillna(1.0).values  # default to 1.0 if no MA
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data invalid
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions with volume filter
        long_breakout = close[i] > weekly_high_aligned[i] and vol_ratio[i] > 2.0
        short_breakout = close[i] < weekly_low_aligned[i] and vol_ratio[i] > 2.0
        
        # Exit conditions: return to opposite weekly level (mean reversion)
        long_exit = close[i] < weekly_low_aligned[i]
        short_exit = close[i] > weekly_high_aligned[i]
        
        # Signal logic
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals