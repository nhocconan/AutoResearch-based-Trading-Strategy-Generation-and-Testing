#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Fade_R4S4_Breakout_Volume
6h strategy using daily Camarilla pivot levels with mean-reversion at R3/S3 and breakout continuation at R4/S4.
- Fade at R3/S3: price touches R3/S3 with volume confirmation, expect reversal toward P
- Breakout at R4/S4: price breaks R4/S4 with volume confirmation, expect continuation in breakout direction
- Uses volume confirmation (1.5x 20-period average) to filter false signals
Designed for ~15-25 trades/year per symbol (60-100 total over 4 years)
Works in ranging markets (fade at R3/S3) and trending markets (breakout at R4/S4)
"""

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
    
    # Get daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each daily bar
    # P = (H + L + C) / 3
    # R4 = C + ((H - L) * 1.1/2)
    # R3 = C + ((H - L) * 1.1/4)
    # S3 = C - ((H - L) * 1.1/4)
    # S4 = C - ((H - L) * 1.1/2)
    pivot = (high_1d + low_1d + close_1d) / 3
    r4 = close_1d + ((high_1d - low_1d) * 1.1 / 2)
    r3 = close_1d + ((high_1d - low_1d) * 1.1 / 4)
    s3 = close_1d - ((high_1d - low_1d) * 1.1 / 4)
    s4 = close_1d - ((high_1d - low_1d) * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation: 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > 1.5 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need enough for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Fade at R3: price touches R3 with volume, expect reversal down
            if high[i] >= r3_aligned[i] and vol_confirm[i]:
                signals[i] = -0.25
                position = -1
            # Fade at S3: price touches S3 with volume, expect reversal up
            elif low[i] <= s3_aligned[i] and vol_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Breakout at R4: price breaks R4 with volume, expect continuation up
            elif close[i] > r4_aligned[i] and vol_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Breakdown at S4: price breaks S4 with volume, expect continuation down
            elif close[i] < s4_aligned[i] and vol_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: fade at R3, or breakdown below S4 (failed breakout)
            if high[i] >= r3_aligned[i] and vol_confirm[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            elif close[i] < s4_aligned[i] and vol_confirm[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: fade at S3, or breakout above R4 (failed breakdown)
            if low[i] <= s3_aligned[i] and vol_confirm[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            elif close[i] > r4_aligned[i] and vol_confirm[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3S3_Fade_R4S4_Breakout_Volume"
timeframe = "6h"
leverage = 1.0