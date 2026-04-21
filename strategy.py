#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_VolumeFilter
Hypothesis: Camarilla pivot levels from 1d provide intraday support/resistance. 
Breakout above R1 or below S1 with volume confirmation (current volume > 1.5x average) 
trades in direction of breakout. Works in bull/bear by capturing momentum bursts.
Uses 12h timeframe for lower frequency, reducing trade count and fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Load 1d data once for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day: based on previous day's OHLC
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    r1_1d = close_1d + (high_1d - low_1d) * 1.1 / 12
    s1_1d = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Calculate volume average for confirmation (20-period)
    volume = prices['volume'].values
    vol_mean = np.zeros(n)
    for i in range(20, n):
        vol_mean[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup
        price = prices['close'].iloc[i]
        vol_current = volume[i]
        vol_avg = vol_mean[i]
        
        # Skip if volume data not ready
        if vol_avg == 0 or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        vol_confirm = vol_current > 1.5 * vol_avg
        
        if position == 0:
            # Long: price breaks above R1 with volume
            if price > r1_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume
            elif price < s1_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price drops back below R1 or volatility spike
            if price < r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises back above S1
            if price > s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_VolumeFilter"
timeframe = "12h"
leverage = 1.0