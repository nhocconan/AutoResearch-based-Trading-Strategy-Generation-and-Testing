#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_Volume_Confirmation_v1
Hypothesis: Camarilla pivot levels (R1/S1) from daily timeframe act as strong support/resistance.
Price breaking above R1 or below S1 with volume confirmation indicates institutional interest.
Works in bull via breakouts and bear via reversals at extreme levels. Target: 25-35 trades/year.
"""

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
    
    # Daily Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot and Camarilla levels
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_hl = high_1d - low_1d
    r1 = close_1d + (range_hl * 1.1 / 12)
    s1 = close_1d - (range_hl * 1.1 / 12)
    
    # Align to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_conf = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 40
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(volume_conf[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ok = volume_conf[i]
        
        if position == 0:
            # Long: break above R1 with volume
            if price > r1_aligned[i] and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume
            elif price < s1_aligned[i] and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price returns to pivot or below S1
            if price <= pivot_aligned[i] or price < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price returns to pivot or above R1
            if price >= pivot_aligned[i] or price > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_Volume_Confirmation_v1"
timeframe = "4h"
leverage = 1.0