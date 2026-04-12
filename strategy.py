#!/usr/bin/env python3
"""
12h_1d_camarilla_volume
Uses 12h timeframe with Camarilla pivot levels from 1d timeframe.
Enters long when price touches or breaks above H4 resistance after pullback,
short when touches or breaks below L4 support after pullback.
Uses volume confirmation (1.5x average volume) to avoid false breaks.
Exits when price reaches opposite H3/L3 level or closes back inside H4/L4 range.
Designed for low trade frequency (target: 15-30 trades/year) to minimize fee drag.
Works in trending markets by capturing continuation after pullback to pivot levels.
"""

name = "12h_1d_camarilla_volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period"""
    typical_price = (high + low + close) / 3
    range_val = high - low
    
    H4 = close + range_val * 1.1 / 2
    H3 = close + range_val * 1.1 / 4
    H2 = close + range_val * 1.1 / 6
    H1 = close + range_val * 1.1 / 12
    
    L1 = close - range_val * 1.1 / 12
    L2 = close - range_val * 1.1 / 6
    L3 = close - range_val * 1.1 / 4
    L4 = close - range_val * 1.1 / 2
    
    return H4, H3, H2, H1, L1, L2, L3, L4

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for 1d
    H4_1d, H3_1d, H2_1d, H1_1d, L1_1d, L2_1d, L3_1d, L4_1d = calculate_camarilla(
        high_1d, low_1d, close_1d
    )
    
    # Align Camarilla levels to 12h timeframe
    H4_1d_aligned = align_htf_to_ltf(prices, df_1d, H4_1d)
    H3_1d_aligned = align_htf_to_ltf(prices, df_1d, H3_1d)
    L3_1d_aligned = align_htf_to_ltf(prices, df_1d, L3_1d)
    L4_1d_aligned = align_htf_to_ltf(prices, df_1d, L4_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(H4_1d_aligned[i]) or np.isnan(H3_1d_aligned[i]) or 
            np.isnan(L3_1d_aligned[i]) or np.isnan(L4_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: price touches/breaks H4 after pullback to H3 or below, with volume
        if (close[i] >= H4_1d_aligned[i] and 
            (close[i-1] <= H3_1d_aligned[i] or close[i-1] <= H4_1d_aligned[i]) and
            vol_confirm[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: price touches/breaks L4 after pullback to L3 or above, with volume
        elif (close[i] <= L4_1d_aligned[i] and 
              (close[i-1] >= L3_1d_aligned[i] or close[i-1] >= L4_1d_aligned[i]) and
              vol_confirm[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Long exit: price reaches H3 or closes back below H4
        elif position == 1 and (close[i] <= H3_1d_aligned[i] or close[i] < H4_1d_aligned[i]):
            position = 0
            signals[i] = 0.0
        # Short exit: price reaches L3 or closes back above L4
        elif position == -1 and (close[i] >= L3_1d_aligned[i] or close[i] > L4_1d_aligned[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals