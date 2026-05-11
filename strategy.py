#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike"
timeframe = "4h"
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
    
    # 1d data for Camarilla pivot, EMA34 trend filter, and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Previous day's close for Camarilla calculation
    prev_close_1d = np.concatenate([[np.nan], close_1d[:-1]])
    
    # Calculate Camarilla levels (R1, S1) for each day
    high_low = high_1d - low_1d
    r1_1d = prev_close_1d + high_low * 1.1 / 12
    s1_1d = prev_close_1d - high_low * 1.1 / 12
    
    # Calculate daily EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 20-period average volume for volume spike filter
    avg_vol_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 4h
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    
    # Calculate 4h volume ratio (current volume / 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma  # Current volume relative to recent average
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50  # Ensure EMA34 and volume MA are ready
    
    for i in range(start_idx, n):
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(avg_vol_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with volume spike and above daily EMA34
            if (close[i] > r1_1d_aligned[i] and 
                vol_ratio[i] > 2.0 and 
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume spike and below daily EMA34
            elif (close[i] < s1_1d_aligned[i] and 
                  vol_ratio[i] > 2.0 and 
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price drops below S1 or volume drops
            if (close[i] < s1_1d_aligned[i] or 
                vol_ratio[i] < 1.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises above R1 or volume drops
            if (close[i] > r1_1d_aligned[i] or 
                vol_ratio[i] < 1.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals