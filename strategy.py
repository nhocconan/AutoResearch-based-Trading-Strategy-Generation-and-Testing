#!/usr/bin/env python3
"""
12h_1D_Camarilla_Pivot_Breakout_Volume
Hypothesis: Uses 1-day Camarilla pivot levels (R1, S1) with volume confirmation for breakout trades on 12h timeframe.
Camarilla levels provide high-probability reversal/breakout points based on prior day's range.
Volume confirmation filters false breakouts. Designed for low trade frequency (~15-30/year) to minimize fee drag.
Works in bull markets (breakouts continue) and bear markets (fades at S1/R1).
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
    
    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels: R1, S1 based on prior day's range
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    R1 = np.full(len(high_1d), np.nan)
    S1 = np.full(len(high_1d), np.nan)
    
    for i in range(1, len(high_1d)):  # Start from 1 to use previous day's data
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        range_val = prev_high - prev_low
        R1[i] = prev_close + 1.1 * range_val / 12
        S1[i] = prev_close - 1.1 * range_val / 12
    
    # Volume spike: current volume > 2.0 x 24-period average (24*12h = 12 days)
    vol_ma = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma[i] = np.mean(volume[i-24:i])
    vol_spike = volume > (vol_ma * 2.0)
    
    # Align 1d Camarilla levels to 12h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(24, 1)  # Need volume MA and at least 2 days of 1d data
    
    for i in range(start_idx, n):
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above R1 with volume spike
            if close[i] > R1_aligned[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume spike
            elif close[i] < S1_aligned[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: close back below R1 (mean reversion) or stop loss
            if close[i] < R1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close back above S1 (mean reversion) or stop loss
            if close[i] > S1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1D_Camarilla_Pivot_Breakout_Volume"
timeframe = "12h"
leverage = 1.0