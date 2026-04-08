#!/usr/bin/env python3
# 4h_1d_cam_pivot_volume_v1
# Hypothesis: Use 1d Camarilla pivot levels with volume confirmation and 4h EMA trend filter.
# Long when price touches S1 support in a rising 4h EMA trend with volume spike.
# Short when price touches R1 resistance in a falling 4h EMA trend with volume spike.
# Exit when price crosses the pivot point (PP) or reverses at opposite Camarilla level.
# Target: 20-40 trades/year with strict entry conditions.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_cam_pivot_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: PP, S1, S2, S3, S4, R1, R2, R3, R4
    # PP = (H + L + C) / 3
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    # R1 = C + (H - L) * 1.1 / 12
    r1_1d = close_1d + (high_1d - low_1d) * 1.1 / 12.0
    # S1 = C - (H - L) * 1.1 / 12
    s1_1d = close_1d - (high_1d - low_1d) * 1.1 / 12.0
    # R2 = C + (H - L) * 1.1 / 6
    r2_1d = close_1d + (high_1d - low_1d) * 1.1 / 6.0
    # S2 = C - (H - L) * 1.1 / 6
    s2_1d = close_1d - (high_1d - low_1d) * 1.1 / 6.0
    # R3 = C + (H - L) * 1.1 / 4
    r3_1d = close_1d + (high_1d - low_1d) * 1.1 / 4.0
    # S3 = C - (H - L) * 1.1 / 4
    s3_1d = close_1d - (high_1d - low_1d) * 1.1 / 4.0
    # R4 = C + (H - L) * 1.1 / 2
    r4_1d = close_1d + (high_1d - low_1d) * 1.1 / 2.0
    # S4 = C - (H - L) * 1.1 / 2
    s4_1d = close_1d - (high_1d - low_1d) * 1.1 / 2.0
    
    # Align Camarilla levels to 4h timeframe
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # 4h EMA trend filter (21-period)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_4h_21 = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_4h_21_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_21)
    
    # Volume filter: 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_4h_21_aligned[i]) or np.isnan(pp_1d_aligned[i]) or 
            np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 2.0 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: Price crosses below PP or touches R1 resistance
            if close[i] < pp_1d_aligned[i] or close[i] >= r1_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price crosses above PP or touches S1 support
            if close[i] > pp_1d_aligned[i] or close[i] <= s1_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price touches S1 support, 4h EMA rising, volume surge
            if (abs(close[i] - s1_1d_aligned[i]) < 0.001 * close[i] and  # Within 0.1% of S1
                ema_4h_21_aligned[i] > ema_4h_21_aligned[i-1] and 
                vol_surge):
                position = 1
                signals[i] = 0.25
            # Short entry: Price touches R1 resistance, 4h EMA falling, volume surge
            elif (abs(close[i] - r1_1d_aligned[i]) < 0.001 * close[i] and  # Within 0.1% of R1
                  ema_4h_21_aligned[i] < ema_4h_21_aligned[i-1] and 
                  vol_surge):
                position = -1
                signals[i] = -0.25
    
    return signals