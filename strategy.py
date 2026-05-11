#!/usr/bin/env python3
name = "1h_Camarilla_R1S1_Breakout_4hTrend_1dVolume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend (Camarilla pivot levels)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 10:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels on 4h
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    pivot_4h = (high_4h + low_4h + close_4h) / 3.0
    range_4h = high_4h - low_4h
    r1_4h = close_4h + (range_4h * 1.1 / 12)
    s1_4h = close_4h - (range_4h * 1.1 / 12)
    
    # Align 4h Camarilla levels to 1h
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 20-period average volume on 1d
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not session_mask[i]:
            if position == 1:
                signals[i] = 0.0
                position = 0
            elif position == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(r1_4h_aligned[i]) or np.isnan(s1_4h_aligned[i]) or 
            np.isnan(vol_1d_aligned[i])):
            if position == 1:
                signals[i] = 0.0
                position = 0
            elif position == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter: current 1h volume > 1.5x 1d average volume
        volume_surge = volume[i] > (vol_1d_aligned[i] * 1.5)
        
        if position == 0:
            # Long: Price breaks above R1 with volume surge
            if close[i] > r1_4h_aligned[i] and volume_surge:
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below S1 with volume surge
            elif close[i] < s1_4h_aligned[i] and volume_surge:
                signals[i] = -0.20
                position = -1
        else:
            # Exit: price returns to pivot level
            pivot_4h_aligned = align_htf_to_ltf(prices, df_4h, 
                                               (df_4h['high'].values + df_4h['low'].values + df_4h['close'].values) / 3.0)
            if np.isnan(pivot_4h_aligned[i]):
                pivot_4h_aligned_i = pivot_4h_aligned[i-1] if i > 0 else 0
            else:
                pivot_4h_aligned_i = pivot_4h_aligned[i]
            
            if position == 1:
                # Exit long: price returns to pivot
                if close[i] <= pivot_4h_aligned_i:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            elif position == -1:
                # Exit short: price returns to pivot
                if close[i] >= pivot_4h_aligned_i:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals