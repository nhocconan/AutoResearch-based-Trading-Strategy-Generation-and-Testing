#!/usr/bin/env python3
"""
4h_1d_Camarilla_Pivot_R1S1_Breakout_Volume_Tight
Hypothesis: Uses daily Camarilla pivot levels R1/S1 as entry triggers with volume confirmation and a 2-bar hold requirement.
Reduces overtrading by requiring price to close outside the level for 2 consecutive bars before entering.
Targets 20-30 trades/year to minimize fee drift. Works in both bull and bear markets by trading breakouts of key daily support/resistance.
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
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels for each day
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    rng = high_1d - low_1d
    r1 = close_1d + rng * 1.1 / 12
    s1 = close_1d - rng * 1.1 / 12
    
    # Align daily levels to 4h timeframe (wait for daily close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    consecutive_outside = 0  # count of consecutive bars outside the level
    
    start_idx = 20
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            consecutive_outside = 0
            continue
        
        # Check if price is outside R1 or S1
        outside_r1 = close[i] > r1_aligned[i]
        outside_s1 = close[i] < s1_aligned[i]
        
        if position == 0:
            # Require 2 consecutive closes outside the level to enter
            if outside_r1:
                consecutive_outside += 1
                if consecutive_outside >= 2 and vol_confirm[i]:
                    signals[i] = 0.25
                    position = 1
                    consecutive_outside = 0  # reset after entry
                else:
                    signals[i] = 0.0
            elif outside_s1:
                consecutive_outside += 1
                if consecutive_outside >= 2 and vol_confirm[i]:
                    signals[i] = -0.25
                    position = -1
                    consecutive_outside = 0  # reset after entry
                else:
                    signals[i] = 0.0
            else:
                consecutive_outside = 0
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price returns to daily pivot or breaks below S1
            pivot_1d = (high_1d + low_1d + close_1d) / 3
            pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
            if not np.isnan(pivot_aligned[i]) and close[i] < pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
                consecutive_outside = 0
            else:
                signals[i] = 0.25
                # Reset counter if still outside R1
                if outside_r1:
                    consecutive_outside = 0
                else:
                    consecutive_outside += 1
        
        elif position == -1:
            # Short exit: price returns to daily pivot or breaks above R1
            pivot_1d = (high_1d + low_1d + close_1d) / 3
            pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
            if not np.isnan(pivot_aligned[i]) and close[i] > pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
                consecutive_outside = 0
            else:
                signals[i] = -0.25
                # Reset counter if still outside S1
                if outside_s1:
                    consecutive_outside = 0
                else:
                    consecutive_outside += 1
    
    return signals

name = "4h_1d_Camarilla_Pivot_R1S1_Breakout_Volume_Tight"
timeframe = "4h"
leverage = 1.0