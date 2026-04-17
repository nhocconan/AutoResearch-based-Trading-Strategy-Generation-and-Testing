#!/usr/bin/env python3
"""
1d_1w_Camarilla_Pivot_R1S1_Breakout_VolumeConfirm_v1
Daily strategy using 1-week Camarilla pivot levels (R1/S1) with volume confirmation.
Enters long when price breaks above R1 with volume above average.
Enters short when price breaks below S1 with volume above average.
Uses tight entry conditions to limit trades and avoid fee drag.
Works in both bull and bear markets by trading breakouts with volume confirmation.
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
    
    # === 1-week Camarilla Pivot Levels (R1, S1) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate pivot point and Camarilla levels
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = close_1w + (high_1w - low_1w) * 1.1 / 12
    s1_1w = close_1w - (high_1w - low_1w) * 1.1 / 12
    
    # Align to daily timeframe
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # === 1-week Volume for Confirmation ===
    volume_1w = df_1w['volume'].values
    vol_ma_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 60
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or 
            np.isnan(vol_ma_1w_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get current 1w bar's volume for confirmation
        vol_1w_current = align_htf_to_ltf(prices, df_1w, volume_1w)[i]
        vol_confirmed = vol_1w_current > 1.5 * vol_ma_1w_aligned[i]
        
        # Breakout conditions
        breakout_long = close[i] > r1_1w_aligned[i]
        breakout_short = close[i] < s1_1w_aligned[i]
        
        # Exit conditions: return to opposite pivot level
        exit_long = close[i] < s1_1w_aligned[i]
        exit_short = close[i] > r1_1w_aligned[i]
        
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

name = "1d_1w_Camarilla_Pivot_R1S1_Breakout_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0