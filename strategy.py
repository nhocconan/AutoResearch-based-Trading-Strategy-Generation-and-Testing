#!/usr/bin/env python3
"""
4h_12h_Camarilla_Pivot_Breakout_Volume
Hypothesis: Use Camarilla pivot levels from 12h as primary price channels (R1,S1) with breakout logic. Add volume confirmation (>1.5x 20-period average) to filter false breakouts. Works in bull markets by buying breaks above R1 with volume, and in bear markets by selling breaks below S1 with volume. The 12h timeframe reduces noise vs lower timeframes while capturing multi-day moves. Targets ~25-35 trades/year by requiring both pivot breakout and volume confirmation, avoiding overtrading.
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
    
    # Get 12h data for Camarilla pivot calculation
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels for each 12h bar
    # R1 = close + (high - low) * 1.1/12
    # S1 = close - (high - low) * 1.1/12
    range_12h = high_12h - low_12h
    r1_12h = close_12h + range_12h * 1.1 / 12
    s1_12h = close_12h - range_12h * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe (wait for 12h bar close)
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    
    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_12h_aligned[i]) or np.isnan(s1_12h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above R1 with volume confirmation
            if close[i] > r1_12h_aligned[i] and vol_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S1 with volume confirmation
            elif close[i] < s1_12h_aligned[i] and vol_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price returns below S1 (mean reversion) or breaks below R1 (failed breakout)
            if close[i] < s1_12h_aligned[i] or close[i] < r1_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above R1 (mean reversion) or breaks above S1 (failed breakout)
            if close[i] > r1_12h_aligned[i] or close[i] > s1_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_12h_Camarilla_Pivot_Breakout_Volume"
timeframe = "4h"
leverage = 1.0