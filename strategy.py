#!/usr/bin/env python3
"""
4h_Wick_Reversal_AntiTrend
Hypothesis: Price reversals at daily high/low with long wicks indicate exhaustion in both bull and bear markets. Combines 1d high/low with 4h wick rejection and volume confirmation for high-probability reversals. Low trade frequency via strict wick and volume filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for daily high/low
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d high and low
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Align 1d high/low to 4h timeframe (using previous day's close for alignment)
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    
    # Wick analysis: long upper/lower shadow
    body_size = np.abs(close - np.minimum(open := prices['open'].values, close))
    upper_wick = high - np.maximum(open, close)
    lower_wick = np.minimum(open, close) - low
    # Long wick condition: wick > 2x body size
    long_upper_wick = upper_wick > (2 * body_size + 1e-10)
    long_lower_wick = lower_wick > (2 * body_size + 1e-10)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[0:i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-20+1:i+1])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = 20  # Warmup for volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(high_1d_aligned[i]) or np.isnan(low_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Long: rejection at daily low with long lower wick and volume spike
            if (low[i] <= low_1d_aligned[i] * 1.001 and  # near daily low
                long_lower_wick[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: rejection at daily high with long upper wick and volume spike
            elif (high[i] >= high_1d_aligned[i] * 0.999 and  # near daily high
                  long_upper_wick[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        
        elif position == 1:
            # Exit: hold max 8 bars or reverse signal
            if bars_since_entry >= 8:
                signals[i] = 0.0
                position = 0
            elif long_upper_wick[i] and vol_spike[i]:  # reverse signal
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # hold
        
        elif position == -1:
            # Exit: hold max 8 bars or reverse signal
            if bars_since_entry >= 8:
                signals[i] = 0.0
                position = 0
            elif long_lower_wick[i] and vol_spike[i]:  # reverse signal
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # hold
    
    return signals

name = "4h_Wick_Reversal_AntiTrend"
timeframe = "4h"
leverage = 1.0