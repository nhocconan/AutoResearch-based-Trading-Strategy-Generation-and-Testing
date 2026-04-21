#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_VolumeFilter_Tight
Hypothesis: Camarilla pivot levels (R1, S1) from 1d timeframe act as key support/resistance. 
Breakout above R1 or below S1 with volume confirmation (volume > 1.5x 20-period average) 
provides high-probability trend continuation. Tight filters to limit trades and reduce fee drag.
Works in bull/bear by taking breakouts in direction of intraday momentum.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period"""
    typical = (high + low + close) / 3
    range_val = high - low
    
    # Camarilla levels
    R4 = close + range_val * 1.500
    R3 = close + range_val * 1.250
    R2 = close + range_val * 1.166
    R1 = close + range_val * 1.083
    S1 = close - range_val * 1.083
    S2 = close - range_val * 1.166
    S3 = close - range_val * 1.250
    S4 = close - range_val * 1.500
    
    return R1, R2, R3, R4, S1, S2, S3, S4

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Load 1d data once for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    R1_1d, _, _, _, S1_1d, _, _, _ = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Align Camarilla levels to 4h timeframe
    R1_1d_aligned = align_htf_to_ltf(prices, df_1d, R1_1d)
    S1_1d_aligned = align_htf_to_ltf(prices, df_1d, S1_1d)
    
    # 4h price and volume data
    close_4h = prices['close'].values
    volume_4h = prices['volume'].values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = np.zeros_like(volume_4h)
    for i in range(len(volume_4h)):
        if i < 19:
            vol_ma[i] = np.nan
        else:
            vol_ma[i] = np.mean(volume_4h[i-19:i+1])
    volume_filter = volume_4h > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if NaN in critical values
        if np.isnan(R1_1d_aligned[i]) or np.isnan(S1_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_4h[i]
        r1 = R1_1d_aligned[i]
        s1 = S1_1d_aligned[i]
        vol_ok = volume_filter[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume confirmation
            if price > r1 and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume confirmation
            elif price < s1 and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price drops back below R1
            if price < r1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises back above S1
            if price > s1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_VolumeFilter_Tight"
timeframe = "4h"
leverage = 1.0