#!/usr/bin/env python3
name = "12h_Camarilla_Pivot_S1S3_Breakout_1wTrend_Volume"
timeframe = "12h"
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
    
    # Get weekly data for trend (1w)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Weekly EMA 50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily Camarilla pivot levels (S1, S3, R1, R3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r3 = pivot + (range_1d * 1.1 / 2)  # R3
    r1 = pivot + (range_1d * 1.1 / 4)  # R1
    s1 = pivot - (range_1d * 1.1 / 4)  # S1
    s3 = pivot - (range_1d * 1.1 / 2)  # S3
    
    # Align pivots to 12h
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume filter: 20-period average on 12h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume threshold - avoid low-volume false breakouts
        volume_surge = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: Price breaks above R1 with volume AND weekly trend up
            if (close[i] > r1_aligned[i] and 
                volume_surge and 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with volume AND weekly trend down
            elif (close[i] < s1_aligned[i] and 
                  volume_surge and 
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price reaches opposite extreme (S3 for long, R3 for short) or returns to pivot
            if position == 1:
                # Exit long: price reaches S3 or returns to pivot
                if (close[i] < s3_aligned[i]) or (close[i] < pivot[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price reaches R3 or returns to pivot
                if (close[i] > r3_aligned[i]) or (close[i] > pivot[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals