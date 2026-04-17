#!/usr/bin/env python3
"""
4h_Pivot_R1_S1_Breakout_Volume_Filter_v1
Breakout above Camarilla R1 or below S1 with volume confirmation.
Uses 1d timeframe for Camarilla levels and 1h for volume spike filter.
Long when price breaks above R1 with volume > 1.5x 20-period average.
Short when price breaks below S1 with volume > 1.5x 20-period average.
Exit when price returns to Pivot point (PP) or reverses with volume confirmation.
Designed to capture institutional breakouts with volume confirmation.
Target: 50-150 total trades over 4 years (12-37/year).
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
    
    # === Volume spike filter (1h timeframe) ===
    df_1h = get_htf_data(prices, '1h')
    vol_ma = pd.Series(df_1h['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1h, vol_ma)
    
    # === Camarilla pivot levels (1d timeframe) ===
    df_1d = get_htf_data(prices, '1d')
    # Calculate Camarilla levels from previous day's OHLC
    # H, L, C from previous day
    phigh = df_1d['high'].values
    plow = df_1d['low'].values
    pclose = df_1d['close'].values
    
    # Camarilla formulas
    # R4 = C + ((H-L) * 1.1/2)
    # R3 = C + ((H-L) * 1.1/4)
    # R2 = C + ((H-L) * 1.1/6)
    # R1 = C + ((H-L) * 1.1/12)
    # PP = (H + L + C) / 3
    # S1 = C - ((H-L) * 1.1/12)
    # S2 = C - ((H-L) * 1.1/6)
    # S3 = C - ((H-L) * 1.1/4)
    # S4 = C - ((H-L) * 1.1/2)
    
    rng = phigh - plow
    r1 = pclose + (rng * 1.1 / 12)
    s1 = pclose - (rng * 1.1 / 12)
    pp = (phigh + plow + pclose) / 3.0
    
    # Align to LTF
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(pp_aligned[i]) or 
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume spike condition: current volume > 1.5x 20-period average
        vol_spike = volume[i] > (1.5 * vol_ma_aligned[i])
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above R1 with volume spike
            if (close[i] > r1_aligned[i] and vol_spike):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below S1 with volume spike
            elif (close[i] < s1_aligned[i] and vol_spike):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price returns to PP or reverses below R1 with volume
            if (close[i] <= pp_aligned[i] or 
                (close[i] < r1_aligned[i] and vol_spike)):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to PP or reverses above S1 with volume
            if (close[i] >= pp_aligned[i] or 
                (close[i] > s1_aligned[i] and vol_spike)):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Pivot_R1_S1_Breakout_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0