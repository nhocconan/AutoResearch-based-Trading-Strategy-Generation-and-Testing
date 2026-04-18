#!/usr/bin/env python3
"""
1d_1w_Camarilla_Pivot_R1S1_Breakout_Volume
Hypothesis: Breakout of weekly Camarilla R1/S1 levels with volume confirmation on daily timeframe.
Trades in the direction of weekly price position relative to weekly pivot to avoid counter-trend trades.
Targets 10-25 trades per year by using strict weekly pivot levels and volume confirmation.
Works in both bull and bear markets by following weekly trend bias.
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
    
    # Get weekly data for pivot levels (HTF)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly Camarilla levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    rng_1w = high_1w - low_1w
    r1_1w = close_1w + rng_1w * 1.1 / 12
    s1_1w = close_1w - rng_1w * 1.1 / 12
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    
    # Align all levels to daily timeframe (wait for bar close)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    
    # Volume confirmation: current volume > 2.0 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) or 
            np.isnan(pivot_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above weekly R1, above weekly pivot, with volume
            if (close[i] > r1_1w_aligned[i] and 
                close[i] > pivot_1w_aligned[i] and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below weekly S1, below weekly pivot, with volume
            elif (close[i] < s1_1w_aligned[i] and 
                  close[i] < pivot_1w_aligned[i] and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price returns to weekly S1 or below weekly pivot
            if (not np.isnan(s1_1w_aligned[i]) and close[i] < s1_1w_aligned[i]) or \
               (close[i] < pivot_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to weekly R1 or above weekly pivot
            if (not np.isnan(r1_1w_aligned[i]) and close[i] > r1_1w_aligned[i]) or \
               (close[i] > pivot_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_Camarilla_Pivot_R1S1_Breakout_Volume"
timeframe = "1d"
leverage = 1.0