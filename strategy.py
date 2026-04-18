#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_Breakout_Volume
Hypothesis: Camarilla pivot levels (R1/S1) from daily timeframe provide significant support/resistance levels.
Breakout above R1 with volume confirmation triggers long, breakdown below S1 with volume triggers short.
Uses 12h as primary timeframe for lower frequency, and 1d for pivot calculation (proven to reduce whipsaws).
Targets 15-25 trades/year by requiring pivot breakout + volume > 2x 20-period average.
Works in bull markets by buying breakouts above daily resistance, and in bear markets by selling breakdowns below daily support.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation (HTF)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each day
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R1 = C + (Range * 1.1 / 12)
    # S1 = C - (Range * 1.1 / 12)
    pivot = (high_1d + low_1d + close_1d) / 3.0
    rng = high_1d - low_1d
    r1 = close_1d + (rng * 1.1 / 12.0)
    s1 = close_1d - (rng * 1.1 / 12.0)
    
    # Align pivot levels to 12h timeframe (wait for bar close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume confirmation: current volume > 2.0 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above R1, with volume
            if close[i] > r1_aligned[i] and vol_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S1, with volume
            elif close[i] < s1_aligned[i] and vol_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price returns below pivot (mean reversion) or fails to hold above R1
            if close[i] < pivot[i] or close[i] < r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above pivot (mean reversion) or fails to hold below S1
            if close[i] > pivot[i] or close[i] > s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_Pivot_Breakout_Volume"
timeframe = "12h"
leverage = 1.0