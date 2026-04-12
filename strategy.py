#!/usr/bin/env python3
"""
6h_1w_pivot_trend_v1
Hypothesis: Use 1-week pivot points (PP, R1, S1) to determine trend direction on 6h chart.
In bull markets, price stays above weekly PP; in bear markets, below weekly PP.
Enter long when 6h close crosses above weekly R1 with volume confirmation.
Enter short when 6h close crosses below weekly S1 with volume confirmation.
Exit on opposite cross or trend reversal.
Designed for 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
"""

name = "6h_1w_pivot_trend_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Previous 1w bar's pivot points
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w = np.roll(close_1w, 1)
    
    pivot = (prev_high_1w + prev_low_1w + prev_close_1w) / 3.0
    r1 = 2 * pivot - prev_low_1w
    s1 = 2 * pivot - prev_high_1w
    
    # Align pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: close crosses above R1 with volume
        if (close[i] > r1_aligned[i] and close[i-1] <= r1_aligned[i-1] and vol_confirm[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: close crosses below S1 with volume
        elif (close[i] < s1_aligned[i] and close[i-1] >= s1_aligned[i-1] and vol_confirm[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: reverse signal or close crosses back to opposite level
        elif position == 1 and close[i] < s1_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > r1_aligned[i]:
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