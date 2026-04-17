#!/usr/bin/env python3
"""
4h_Pivot_R1_S1_Breakout_Volume_Spike_v2
Daily pivot points R1/S1 breakout with volume spike confirmation.
Long when price breaks above R1 with volume > 2x MA(20), short when breaks below S1 with volume spike.
Exit when price returns to pivot point (PP) or reverses with volume confirmation.
Uses 1d timeframe for pivot calculation and volume confirmation.
Target: 20-50 trades/year per symbol.
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
    
    # === Daily Pivot Points ===
    df_1d = get_htf_data(prices, '1d')
    # Calculate pivot points from previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot Point (PP) = (H + L + C) / 3
    pp = (high_1d + low_1d + close_1d) / 3.0
    # Resistance 1 (R1) = (2 * PP) - L
    r1 = (2 * pp) - low_1d
    # Support 1 (S1) = (2 * PP) - H
    s1 = (2 * pp) - high_1d
    
    # Align to 4h timeframe (use previous day's levels)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === Volume Spike Filter ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma)  # Volume at least 2x 20-period average
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 30
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above R1 with volume spike
            if (close[i] > r1_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below S1 with volume spike
            elif (close[i] < s1_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price returns to PP or breaks below S1 with volume
            if (close[i] <= pp_aligned[i] or 
                (close[i] < s1_aligned[i] and vol_spike[i])):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to PP or breaks above R1 with volume
            if (close[i] >= pp_aligned[i] or 
                (close[i] > r1_aligned[i] and vol_spike[i])):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Pivot_R1_S1_Breakout_Volume_Spike_v2"
timeframe = "4h"
leverage = 1.0