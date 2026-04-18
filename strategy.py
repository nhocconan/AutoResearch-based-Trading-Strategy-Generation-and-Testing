#!/usr/bin/env python3
"""
12h_1D_Camarilla_Pivot_Breakout_Volume
Hypothesis: Uses 1d Camarilla pivot levels (R1/S1) for breakout entries on 12h timeframe with volume confirmation.
Camarilla levels provide statistically significant support/resistance in ranging and trending markets.
Volume filter ensures breakouts have conviction. Targets 15-25 trades/year per symbol.
Works in bull/bear via mean-reversion at extremes and breakout continuation.
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
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla multipliers
    R1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    S1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    R2 = close_1d + (high_1d - low_1d) * 1.1 / 6
    S2 = close_1d - (high_1d - low_1d) * 1.1 / 6
    
    # Volume spike: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    # Align 1d levels to 12h timeframe
    R1_12h = align_htf_to_ltf(prices, df_1d, R1)
    S1_12h = align_htf_to_ltf(prices, df_1d, S1)
    R2_12h = align_htf_to_ltf(prices, df_1d, R2)
    S2_12h = align_htf_to_ltf(prices, df_1d, S2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        if (np.isnan(R1_12h[i]) or np.isnan(S1_12h[i]) or 
            np.isnan(R2_12h[i]) or np.isnan(S2_12h[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above R1 with volume
            if close[i] > R1_12h[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume
            elif close[i] < S1_12h[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: close below S1 (mean reversion) or break R2 (take profit)
            if close[i] < S1_12h[i] or close[i] > R2_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above R1 (mean reversion) or break S2 (take profit)
            if close[i] > R1_12h[i] or close[i] < S2_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1D_Camarilla_Pivot_Breakout_Volume"
timeframe = "12h"
leverage = 1.0