#!/usr/bin/env python3
"""
6h_12h_Camarilla_Pivot_R4S4_Breakout_Volume
Hypothesis: Use 12h Camarilla pivot levels (R4/S4) as breakout levels with volume confirmation.
In trending markets: breakout above R4 or below S4 signals strong momentum.
In ranging markets: fewer false breakouts due to wider bands and volume filter.
Targets 20-50 trades/year with position size 0.25 to balance opportunity and drawdown.
Works in both bull/bear via breakout logic (direction agnostic).
"""

name = "6h_12h_Camarilla_Pivot_R4S4_Breakout_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Camarilla pivots (based on previous day)
    # Typical price = (H + L + C) / 3
    typical_price = (high_12h + low_12h + close_12h) / 3
    range_12h = high_12h - low_12h
    
    # Camarilla levels: R4 = C + (H-L)*1.1/2, S4 = C - (H-L)*1.1/2
    r4 = typical_price + range_12h * 1.1 / 2
    s4 = typical_price - range_12h * 1.1 / 2
    
    # Align to 6h timeframe (wait for 12h bar to close)
    r4_aligned = align_htf_to_ltf(prices, df_12h, r4)
    s4_aligned = align_htf_to_ltf(prices, df_12h, s4)
    
    # Volume average (20-period) for confirmation
    vol_avg = np.zeros_like(volume)
    vol_avg[:] = np.nan
    for i in range(20, len(volume)):
        vol_avg[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Need 12h data + vol avg
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(vol_avg[i]) or np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout: price crosses above R4 with volume > 1.5x average
            if close[i] > r4_aligned[i] and volume[i] > vol_avg[i] * 1.5:
                signals[i] = 0.25
                position = 1
            # Short breakout: price crosses below S4 with volume > 1.5x average
            elif close[i] < s4_aligned[i] and volume[i] > vol_avg[i] * 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns below R4 (failed breakout) or reverse below S4
            if close[i] < r4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above S4 (failed breakdown) or reverse above R4
            if close[i] > s4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals