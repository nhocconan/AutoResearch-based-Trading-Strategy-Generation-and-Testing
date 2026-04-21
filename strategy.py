#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_VolumeFilter_Tight
Hypothesis: Camarilla pivot levels (R1/S1) from 1d timeframe act as key intraday support/resistance. 
Breakout above R1 or below S1 with volume confirmation on 12h provides high-probability trend continuation. 
Volume filter reduces false breakouts. Works in bull/bear by taking both long and short breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """
    Calculate Camarilla pivot levels:
    P = (High + Low + Close) / 3
    R1 = Close + (High - Low) * 1.1 / 12
    S1 = Close - (High - Low) * 1.1 / 12
    """
    typical = (high + low + close) / 3
    range_val = high - low
    R1 = close + range_val * 1.1 / 12
    S1 = close - range_val * 1.1 / 12
    return R1, S1

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Load 1d data once for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    R1_1d, S1_1d = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Align Camarilla levels to 12h timeframe
    R1_1d_aligned = align_htf_to_ltf(prices, df_1d, R1_1d)
    S1_1d_aligned = align_htf_to_ltf(prices, df_1d, S1_1d)
    
    # 12h volume average for confirmation
    volume_12h = prices['volume'].values
    vol_avg = np.zeros_like(volume_12h)
    for i in range(len(volume_12h)):
        start = max(0, i - 19)  # 20-period average
        vol_avg[i] = np.mean(volume_12h[start:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if NaN in critical values
        if np.isnan(R1_1d_aligned[i]) or np.isnan(S1_1d_aligned[i]) or np.isnan(vol_avg[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = volume_12h[i]
        R1 = R1_1d_aligned[i]
        S1 = S1_1d_aligned[i]
        vol_average = vol_avg[i]
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirm = volume > 1.5 * vol_average
        
        if position == 0:
            # Long: price breaks above R1 with volume confirmation
            if price > R1 and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume confirmation
            elif price < S1 and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price drops below S1 (reversal signal)
            if price < S1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above R1 (reversal signal)
            if price > R1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_VolumeFilter_Tight"
timeframe = "12h"
leverage = 1.0