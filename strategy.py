#!/usr/bin/env python3
"""
4h_1w_HTF_Camarilla_Pivot_S1S4_Breakout
Hypothesis: Uses weekly S1/S4 levels from 1w chart for directional bias and 4h for entry timing.
Trades breakouts of key weekly support/resistance with volume confirmation to reduce false signals.
Designed to work in both bull and bear markets by capturing directional moves after consolidation.
Target: 15-25 trades/year to minimize fee drag while maintaining edge.
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
    
    # Get weekly data for multi-timeframe analysis
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly Camarilla levels (S1 and S4 for strong support/resistance)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    rng_1w = high_1w - low_1w
    s1_1w = close_1w - rng_1w * 1.1 / 12
    s4_1w = close_1w - rng_1w * 1.1 / 2
    
    # Align weekly levels to 4h timeframe (wait for weekly bar close)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    # Volume confirmation: current volume > 1.8 x 24-period average (48h lookback)
    vol_ma = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma[i] = np.mean(volume[i-24:i])
    vol_confirm = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 24
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(s1_1w_aligned[i]) or np.isnan(s4_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above weekly S1 with volume
            if close[i] > s1_1w_aligned[i] and vol_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below weekly S4 with volume
            elif close[i] < s4_1w_aligned[i] and vol_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price returns to weekly S1 or breaks below S4
            if (not np.isnan(s1_1w_aligned[i]) and close[i] < s1_1w_aligned[i]) or \
               (not np.isnan(s4_1w_aligned[i]) and close[i] < s4_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to weekly S4 or breaks above S1
            if (not np.isnan(s4_1w_aligned[i]) and close[i] > s4_1w_aligned[i]) or \
               (not np.isnan(s1_1w_aligned[i]) and close[i] > s1_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1w_HTF_Camarilla_Pivot_S1S4_Breakout"
timeframe = "4h"
leverage = 1.0