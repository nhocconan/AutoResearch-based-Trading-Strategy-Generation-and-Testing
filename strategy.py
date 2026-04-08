#!/usr/bin/env python3
# 6h_weekly_pivot_volume_breakout_v1
# Hypothesis: 6h price breaks above/below weekly Camarilla pivot R4/S4 levels with volume > 2.0x average.
# Long: close > weekly R4 and volume > 2.0x 20-period average
# Short: close < weekly S4 and volume > 2.0x 20-period average
# Exit: price returns to weekly pivot point (PP) or opposite extreme (R3/S3)
# Uses 6h primary timeframe with 1w HTF for pivot levels to capture institutional breakouts.
# Target: 50-150 trades over 4 years (12-37/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_volume_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate volume ratio (current vs 20-period average)
    vol_sma = np.full(n, np.nan)
    for i in range(20, n):
        vol_sma[i] = np.mean(volume[i-20:i])
    vol_ratio = np.where(vol_sma > 0, volume / vol_sma, 0)
    
    # Get 1w data for weekly Camarilla pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Pivot point (PP) = (H + L + C) / 3
    pp = (high_1w + low_1w + close_1w) / 3.0
    # Range = H - L
    rng = high_1w - low_1w
    
    # Camarilla levels
    r4 = pp + rng * 1.1 / 2.0
    r3 = pp + rng * 1.1 / 4.0
    s3 = pp - rng * 1.1 / 4.0
    s4 = pp - rng * 1.1 / 2.0
    
    # Align weekly levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        vol_r = vol_ratio[i]
        price = close[i]
        
        if np.isnan(vol_r):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        pp_val = pp_aligned[i]
        r4_val = r4_aligned[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        s4_val = s4_aligned[i]
        
        if np.isnan(pp_val) or np.isnan(r4_val) or np.isnan(s4_val):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            if price < pp_val or price > r3_val:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            if price > pp_val or price < s3_val:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if price > r4_val and vol_r > 2.0:
                position = 1
                signals[i] = 0.25
            elif price < s4_val and vol_r > 2.0:
                position = -1
                signals[i] = -0.25
    
    return signals