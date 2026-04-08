#!/usr/bin/env python3
"""
12h_1d_camarilla_breakout_volume_v1
Hypothesis: Use 12h price breaks of 1d Camarilla R1/S1 levels with volume confirmation and 1d EMA trend filter. Only trade when price breaks R1 in uptrend (long) or S1 in downtrend (short). Exit when price reverts to pivot or trend changes. Designed for 12h timeframe to target 50-150 total trades over 4 years, avoiding fee drag while working in both bull and bear markets via trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_breakout_volume_v1"
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
    
    # Get 1d data for trend and Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(21) for trend
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point and ranges
    pivot = (high_1d + low_1d + close_1d) / 3
    range_ = high_1d - low_1d
    
    # Key Camarilla levels: R1 and S1 (most significant for intraday)
    r1 = close_1d + range_ * 1.1 / 12
    s1 = close_1d - range_ * 1.1 / 12
    pivot_level = pivot  # for exit
    
    # Align to 12h
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_level)
    
    # Volume confirmation: volume > 1.5x average of last 10 periods (10*12h = 5 days)
    vol_ma = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    vol_confirm = volume > vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to pivot or trend changes
            if close[i] <= pivot_aligned[i] or close[i] < ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price returns to pivot or trend changes
            if close[i] >= pivot_aligned[i] or close[i] > ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price breaks above R1 with volume and uptrend
            if (close[i] > r1_aligned[i] and 
                ema_1d_aligned[i] > ema_1d_aligned[max(0, i-3)] and  # Uptrend confirmation
                vol_confirm[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below S1 with volume and downtrend
            elif (close[i] < s1_aligned[i] and 
                  ema_1d_aligned[i] < ema_1d_aligned[max(0, i-3)] and  # Downtrend confirmation
                  vol_confirm[i]):
                position = -1
                signals[i] = -0.25
    
    return signals