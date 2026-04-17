#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_R1_S1_Breakout_Volume_Spike_v1
Breakout above Camarilla R1 or below S1 with volume spike and volume ratio filter.
Uses 1d timeframe for pivot calculation.
Exit when price returns to H4/L4 levels or volume drops.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Volume spike filter (20-period) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    # === 1d Camarilla pivot levels ===
    df_1d = get_htf_data(prices, '1d')
    # Calculate pivots from previous day's OHLC
    phigh = df_1d['high'].values
    plow = df_1d['low'].values
    pclose = df_1d['close'].values
    
    # Camarilla formula
    range_ = phigh - plow
    R4 = pclose + range_ * 1.1 / 2
    R3 = pclose + range_ * 1.1 / 4
    R2 = pclose + range_ * 1.1 / 6
    R1 = pclose + range_ * 1.1 / 12
    S1 = pclose - range_ * 1.1 / 12
    S2 = pclose - range_ * 1.1 / 6
    S3 = pclose - range_ * 1.1 / 4
    S4 = pclose - range_ * 1.1 / 2
    
    # Align to 4h timeframe (previous day's pivots)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    H4_aligned = align_htf_to_ltf(prices, df_1d, (pclose + range_ * 1.1 / 2) * 0)  # Placeholder, will use R4/S4 for exit
    L4_aligned = align_htf_to_ltf(prices, df_1d, (pclose - range_ * 1.1 / 2) * 0)  # Placeholder
    
    # Actually compute R4 and S4 for exit
    R4_val = pclose + range_ * 1.1 / 2
    S4_val = pclose - range_ * 1.1 / 2
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4_val)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4_val)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 30
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(vol_ratio[i]) or 
            np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or
            np.isnan(R4_aligned[i]) or
            np.isnan(S4_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above R1 with volume spike (vol_ratio > 1.5)
            if (close[i] > R1_aligned[i] and 
                vol_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below S1 with volume spike (vol_ratio > 1.5)
            elif (close[i] < S1_aligned[i] and 
                  vol_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price returns to H4 (R4) level OR volume drops (vol_ratio < 1.0)
            if (close[i] >= R4_aligned[i] or 
                vol_ratio[i] < 1.0):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to L4 (S4) level OR volume drops (vol_ratio < 1.0)
            if (close[i] <= S4_aligned[i] or 
                vol_ratio[i] < 1.0):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_Pivot_R1_S1_Breakout_Volume_Spike_v1"
timeframe = "4h"
leverage = 1.0