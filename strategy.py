#!/usr/bin/env python3
# 12h_daily_camarilla_pivot_volume_v1
# Hypothesis: 12h Camarilla pivot levels from 1d HTF with volume confirmation.
# Long: price breaks above H3 level with volume > 1.5x 20-period average
# Short: price breaks below L3 level with volume > 1.5x 20-period average
# Exit: price reverses to H4/L4 levels or opposite Camarilla level
# Uses 12h primary timeframe with 1d HTF for pivot levels to reduce noise.
# Target: 50-150 trades over 4 years (12-37/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_daily_camarilla_pivot_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h Donchian exit channels (10-period) for stop/reverse
    highest_10 = np.full(n, np.nan)
    lowest_10 = np.full(n, np.nan)
    for i in range(10, n):
        highest_10[i] = np.max(high[i-10:i])
        lowest_10[i] = np.min(low[i-10:i])
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels on 1d data
    # PP = (H + L + C) / 3
    # R4 = PP + (H - L) * 1.1/2
    # R3 = PP + (H - L) * 1.1/4
    # R2 = PP + (H - L) * 1.1/6
    # R1 = PP + (H - L) * 1.1/12
    # S1 = PP - (H - L) * 1.1/12
    # S2 = PP - (H - L) * 1.1/6
    # S3 = PP - (H - L) * 1.1/4
    # S4 = PP - (H - L) * 1.1/2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    r4 = pivot + range_1d * 1.1 / 2.0
    r3 = pivot + range_1d * 1.1 / 4.0
    r2 = pivot + range_1d * 1.1 / 6.0
    r1 = pivot + range_1d * 1.1 / 12.0
    s1 = pivot - range_1d * 1.1 / 12.0
    s2 = pivot - range_1d * 1.1 / 6.0
    s3 = pivot - range_1d * 1.1 / 4.0
    s4 = pivot - range_1d * 1.1 / 2.0
    
    # Align 1d Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate volume ratio (current vs 20-period average)
    vol_sma = np.full(n, np.nan)
    for i in range(20, n):
        vol_sma[i] = np.mean(volume[i-20:i])
    vol_ratio = np.where(vol_sma > 0, volume / vol_sma, 0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        vol_r = vol_ratio[i]
        price = close[i]
        r3 = r3_aligned[i]
        r4 = r4_aligned[i]
        s3 = s3_aligned[i]
        s4 = s4_aligned[i]
        
        if np.isnan(vol_r) or np.isnan(r3) or np.isnan(r4) or np.isnan(s3) or np.isnan(s4):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        exit_high = highest_10[i]
        exit_low = lowest_10[i]
        
        if np.isnan(exit_high) or np.isnan(exit_low):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            if price < exit_low or price > r4:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            if price > exit_high or price < s4:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if price > r3 and vol_r > 1.5:
                position = 1
                signals[i] = 0.25
            elif price < s3 and vol_r > 1.5:
                position = -1
                signals[i] = -0.25
    
    return signals