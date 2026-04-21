#!/usr/bin/env python3
"""
6h_1d_Pivot_MomentumBreakout_WithVolumeConfirmation
Hypothesis: Uses daily pivot points (PP, R1, S1) to identify mean-reversion zones and breakout levels. 
Long when price breaks above R1 with volume confirmation and price > PP (bullish bias). 
Short when price breaks below S1 with volume confirmation and price < PP (bearish bias). 
Exit when price returns to PP or breaks R2/S2 (failed breakout). 
Designed for low trade frequency (target: 12-37/year) to minimize fee drag in 6h timeframe. 
Works in both bull and bear markets by using price relative to PP as bias filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data once for pivot points
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Calculate daily pivot points
    # P = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    # R2 = P + (H - L)
    # S2 = P - (H - L)
    P = (high_daily + low_daily + close_daily) / 3.0
    R1 = 2 * P - low_daily
    S1 = 2 * P - high_daily
    R2 = P + (high_daily - low_daily)
    S2 = P - (high_daily - low_daily)
    
    # Align daily pivot levels to 6h timeframe
    P_aligned = align_htf_to_ltf(prices, df_daily, P)
    R1_aligned = align_htf_to_ltf(prices, df_daily, R1)
    S1_aligned = align_htf_to_ltf(prices, df_daily, S1)
    R2_aligned = align_htf_to_ltf(prices, df_daily, R2)
    S2_aligned = align_htf_to_ltf(prices, df_daily, S2)
    
    # Main timeframe data (6h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.5x 24-period average (4 days of 6h bars)
    volume_avg = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 24:
            volume_avg[i] = np.mean(volume[i-24:i])
        else:
            volume_avg[i] = np.mean(volume[:i+1]) if i > 0 else volume[i]
    volume_filter = volume > (1.5 * volume_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(24, n):
        # Skip if NaN in critical values
        if (np.isnan(P_aligned[i]) or np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(R2_aligned[i]) or np.isnan(S2_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        PP = P_aligned[i]
        R1 = R1_aligned[i]
        S1 = S1_aligned[i]
        R2 = R2_aligned[i]
        S2 = S2_aligned[i]
        vol_ok = volume_filter[i]
        
        if position == 0:
            # Long: break above R1 with volume and bullish bias (price > PP)
            if price > R1 and vol_ok and price > PP:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume and bearish bias (price < PP)
            elif price < S1 and vol_ok and price < PP:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: return to PP (mean reversion) or break R2 (failed breakout)
            if price < PP or price > R2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: return to PP (mean reversion) or break S2 (failed breakdown)
            if price > PP or price < S2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1d_Pivot_MomentumBreakout_WithVolumeConfirmation"
timeframe = "6h"
leverage = 1.0