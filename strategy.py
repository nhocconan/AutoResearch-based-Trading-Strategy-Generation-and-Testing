#!/usr/bin/env python3
"""
6h_Camarilla_R1_S1_Breakout_Volume_Trend_Filter
Hypothesis: On 6h timeframe, enter long when price breaks above Camarilla R1 with daily volume confirmation and daily trend alignment; short when breaks below S1. Uses daily pivot levels to capture institutional levels. Designed for 50-150 total trades over 4 years to minimize fee drag and work in both bull/bear markets via trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Daily data for Camarilla pivot levels and volume ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Daily Camarilla pivot calculation: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    hl_range = high_1d - low_1d
    r1_1d = close_1d + hl_range * 1.1 / 12
    s1_1d = close_1d - hl_range * 1.1 / 12
    
    # Align R1 and S1 to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Daily 20-period average volume for confirmation
    vol_avg20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg20_1d)
    
    # Daily 50-period SMA for trend filter
    sma50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    sma50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma50_1d)
    
    signals = np.zeros(n)
    
    # Warmup: covers 50-day SMA and 20-day volume average
    warmup = 100
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_avg20_1d_aligned[i]) or np.isnan(sma50_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get current daily volume
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        
        # Volume filter: current volume > 1.5x 20-day average
        vol_filter = vol_1d_current > 1.5 * vol_avg20_1d_aligned[i]
        
        # Trend filter: price above/below daily 50 SMA
        above_trend = close[i] > sma50_1d_aligned[i]
        below_trend = close[i] < sma50_1d_aligned[i]
        
        # Entry conditions
        if position == 0:
            # Long: price > Camarilla R1 + volume + above daily trend
            if close[i] > r1_aligned[i] and vol_filter and above_trend:
                signals[i] = 0.25
                position = 1
                continue
            # Short: price < Camarilla S1 + volume + below daily trend
            elif close[i] < s1_aligned[i] and vol_filter and below_trend:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit conditions: reverse signal at opposite Camarilla level
        elif position == 1:
            if close[i] < s1_aligned[i]:  # break below S1 = exit long
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            if close[i] > r1_aligned[i]:  # break above R1 = exit short
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R1_S1_Breakout_Volume_Trend_Filter"
timeframe = "6h"
leverage = 1.0