#!/usr/bin/env python3
"""
4h_1d_Camarilla_Pivot_R1S1_Breakout_VolumeConfirm_v1
4-hour strategy using 1-day Camarilla pivot levels (R1/S1) with volume confirmation.
Enters long when price breaks above R1 with volume above average.
Enters short when price breaks below S1 with volume above average.
Uses tight entry conditions to limit trades and avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 1-day Camarilla Pivot Levels (R1, S1) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point and Camarilla levels
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = close_1d + (high_1d - low_1d) * 1.1 / 12
    s1_1d = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Align to 4h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # === 1-day Volume for Confirmation ===
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 60
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get current 1d bar's volume for confirmation
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        vol_confirmed = vol_1d_current > 1.5 * vol_ma_1d_aligned[i]
        
        # Breakout conditions
        breakout_long = close[i] > r1_1d_aligned[i]
        breakout_short = close[i] < s1_1d_aligned[i]
        
        # Exit conditions: return to opposite pivot level
        exit_long = close[i] < s1_1d_aligned[i]
        exit_short = close[i] > r1_1d_aligned[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: break above R1 with volume confirmation
            if breakout_long and vol_confirmed:
                signals[i] = 0.25
                position = 1
                continue
            # Short: break below S1 with volume confirmation
            elif breakout_short and vol_confirmed:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price breaks below S1
            if exit_long:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above R1
            if exit_short:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_Camarilla_Pivot_R1S1_Breakout_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0