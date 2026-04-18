#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_Volume_Strategy_v1
Hypothesis: Use 12h Camarilla pivot levels (R1, S1) for breakout direction with volume confirmation.
Long when price breaks above 12h R1 with volume > 1.5x 20-period average.
Short when price breaks below 12h S1 with volume > 1.5x 20-period average.
Exit when price returns to the 12h central pivot (PP).
Designed for 4h timeframe to limit trades (target: 20-50/year) and avoid fee drag.
Works in bull markets via breakouts and in bear via mean reversion to pivot.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla pivot levels
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla pivot levels for 12h
    # PP = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    pp_12h = (high_12h + low_12h + close_12h) / 3.0
    r1_12h = close_12h + (high_12h - low_12h) * 1.1 / 12.0
    s1_12h = close_12h - (high_12h - low_12h) * 1.1 / 12.0
    
    # Align Camarilla levels to 4h timeframe
    pp_12h_aligned = align_htf_to_ltf(prices, df_12h, pp_12h)
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(vol_period, 1) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pp_12h_aligned[i]) or np.isnan(r1_12h_aligned[i]) or 
            np.isnan(s1_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above 12h R1 with volume confirmation
            if close[i] > r1_12h_aligned[i] and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12h S1 with volume confirmation
            elif close[i] < s1_12h_aligned[i] and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to or below 12h central pivot (PP)
            if close[i] <= pp_12h_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to or above 12h central pivot (PP)
            if close[i] >= pp_12h_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_Pivot_Volume_Strategy_v1"
timeframe = "4h"
leverage = 1.0