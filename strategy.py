#!/usr/bin/env python3
# 12h_camilla_pivot_breakout_volume_v3
# Hypothesis: Camarilla pivot levels on daily timeframe provide strong support/resistance.
# Price breaking above/below these levels with volume confirmation indicates institutional interest.
# In both bull and bear markets, institutional players defend these levels, creating breakouts.
# Uses 12h timeframe for reduced noise and fewer trades to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camilla_pivot_breakout_volume_v3"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data once before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from previous day
    # Camarilla formulas:
    # H4 = C + ((H-L) * 1.1/2)
    # H3 = C + ((H-L) * 1.1/4)
    # H2 = C + ((H-L) * 1.1/6)
    # H1 = C + ((H-L) * 1.1/12)
    # L1 = C - ((H-L) * 1.1/12)
    # L2 = C - ((H-L) * 1.1/6)
    # L3 = C - ((H-L) * 1.1/4)
    # L4 = C - ((H-L) * 1.1/2)
    # where C = (H+L+CLOSE)/3 of previous day
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point (average of H, L, C)
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Calculate all 8 levels
    H4 = pivot + (range_1d * 1.1 / 2)
    H3 = pivot + (range_1d * 1.1 / 4)
    H2 = pivot + (range_1d * 1.1 / 6)
    H1 = pivot + (range_1d * 1.1 / 12)
    L1 = pivot - (range_1d * 1.1 / 12)
    L2 = pivot - (range_1d * 1.1 / 6)
    L3 = pivot - (range_1d * 1.1 / 4)
    L4 = pivot - (range_1d * 1.1 / 2)
    
    # Align each level to 12h timeframe
    H4_12h = align_htf_to_ltf(prices, df_1d, H4)
    H3_12h = align_htf_to_ltf(prices, df_1d, H3)
    H2_12h = align_htf_to_ltf(prices, df_1d, H2)
    H1_12h = align_htf_to_ltf(prices, df_1d, H1)
    L1_12h = align_htf_to_ltf(prices, df_1d, L1)
    L2_12h = align_htf_to_ltf(prices, df_1d, L2)
    L3_12h = align_htf_to_ltf(prices, df_1d, L3)
    L4_12h = align_htf_to_ltf(prices, df_1d, L4)
    
    # Volume confirmation: 1.5x 24-period average (24 * 12h = 12 days)
    volume = prices['volume'].values
    vol_ma_period = 24
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.5 * vol_ma[i]
    
    close = prices['close'].values
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(24, 1) + 1  # volume MA needs 24 periods
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(H4_12h[i]) or np.isnan(L4_12h[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price below H3 or volume drops below average
            if close[i] < H3_12h[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price above L3 or volume drops below average
            if close[i] > L3_12h[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price above H4 with volume surge
            if (close[i] > H4_12h[i] and vol_surge[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Price below L4 with volume surge
            elif (close[i] < L4_12h[i] and vol_surge[i]):
                position = -1
                signals[i] = -0.25
    
    return signals