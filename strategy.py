#!/usr/bin/env python3
"""
4h_Pivot_R1_S1_Breakout_VolumeSpike_v1
Trades Camarilla pivot levels (R1/S1) from daily high/low/close with volume spike confirmation.
Long when price breaks above R1 with volume > 1.5x 20-period average.
Short when price breaks below S1 with volume > 1.5x 20-period average.
Exit when price returns to daily pivot (PP) or reverses to opposite level.
Uses daily Camarilla levels for structure, 4h for entry timing.
Target: 20-50 trades/year to stay under fee drag threshold.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Daily Camarilla Pivot Levels ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    ph = df_1d['high'].values
    pl = df_1d['low'].values
    pc = df_1d['close'].values
    
    # Pivot point
    pp = (ph + pl + pc) / 3
    # R1 and S1 levels
    r1 = pc + (ph - pl) * 1.1 / 12
    s1 = pc - (ph - pl) * 1.1 / 12
    
    # Align to 4h timeframe (use previous day's levels)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === Volume Spike Filter ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above R1 with volume spike
            if (close[i] > r1_aligned[i] and 
                vol_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below S1 with volume spike
            elif (close[i] < s1_aligned[i] and 
                  vol_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price returns to pivot PP or breaks below S1
            if (close[i] <= pp_aligned[i] or 
                close[i] < s1_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to pivot PP or breaks above R1
            if (close[i] >= pp_aligned[i] or 
                close[i] > r1_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Pivot_R1_S1_Breakout_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0