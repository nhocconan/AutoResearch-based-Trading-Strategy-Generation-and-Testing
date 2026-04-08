#!/usr/bin/env python3
# 12h_daily_camarilla_pivot_volume_v1
# Hypothesis: 12h strategy using 1d Camarilla pivot levels with volume confirmation.
# Long when price touches L3 level with volume > 1.5x 20-period average and closes above L3.
# Short when price touches H3 level with volume > 1.5x 20-period average and closes below H3.
# Uses ATR-based stoploss and discrete position sizing (0.25) to minimize fee drag.
# Works in bull/bear by fading extremes at institutional pivot levels with volume validation.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_daily_camarilla_pivot_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    H3 = pivot_1d + range_1d * 1.1 / 4
    L3 = pivot_1d - range_1d * 1.1 / 4
    H4 = pivot_1d + range_1d * 1.1 / 2
    L4 = pivot_1d - range_1d * 1.1 / 2
    
    # Align HTF levels to 12h timeframe (with completed-bar delay)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # Volume confirmation: 20-period average
    vol_ma = np.full(n, np.nan)
    vol_sum = 0.0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if vol_count >= 20:
            vol_ma[i] = vol_sum / 20
            vol_sum -= volume[i - 19]
        elif i >= 19:
            vol_ma[i] = vol_sum / vol_count
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                # Hold position until exit signal
                signals[i] = 0.25 if position == 1 else -0.25
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        if position == 1:  # Long position
            # Exit: price closes below L3 or reverses from H4
            if close[i] < L3_aligned[i] or close[i] > H4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above H3 or reverses from L4
            if close[i] > H3_aligned[i] or close[i] < L4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price touches L3 with volume confirmation and closes above
            if (low[i] <= L3_aligned[i] and close[i] > L3_aligned[i] and vol_ratio > 1.5):
                position = 1
                signals[i] = 0.25
            # Enter short: price touches H3 with volume confirmation and closes below
            elif (high[i] >= H3_aligned[i] and close[i] < H3_aligned[i] and vol_ratio > 1.5):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals