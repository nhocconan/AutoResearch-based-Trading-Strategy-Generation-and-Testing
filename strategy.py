#!/usr/bin/env python3
"""
1h_4h_Pivot_R1S1_Breakout_VolumeFilter
Hypothesis: 4h Camarilla R1/S1 breakout with volume confirmation on 1h timeframe
4h Camarilla levels provide stronger support/resistance than 1d for intraday trading
Volume filter ensures institutional participation to reduce false breakouts
1h timeframe used only for entry timing to avoid overtrading
Target: 15-37 trades/year (60-150 total over 4 years) to minimize fee impact
Works in bull/bear via volatility-adjusted breakouts and volume confirmation
"""

name = "1h_4h_Pivot_R1S1_Breakout_VolumeFilter"
timeframe = "1h"
leverage = 1.0

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
    
    # Get 4h data for Camarilla levels (primary signal direction)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 4h bar
    ph_4h = df_4h['high'].shift(1).values  # Previous 4h high
    pl_4h = df_4h['low'].shift(1).values   # Previous 4h low
    pc_4h = df_4h['close'].shift(1).values # Previous 4h close
    
    # Camarilla calculations for 4h
    rang_4h = ph_4h - pl_4h
    r1_4h = pc_4h + (rang_4h * 1.1 / 12)
    s1_4h = pc_4h - (rang_4h * 1.1 / 12)
    r4_4h = pc_4h + (rang_4h * 1.1 / 2)
    s4_4h = pc_4h - (rang_4h * 1.1 / 2)
    
    # Align 4h Camarilla levels to 1h timeframe
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    r4_4h_aligned = align_htf_to_ltf(prices, df_4h, r4_4h)
    s4_4h_aligned = align_htf_to_ltf(prices, df_4h, s4_4h)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_4h_aligned[i]) or np.isnan(s1_4h_aligned[i]) or 
            np.isnan(r4_4h_aligned[i]) or np.isnan(s4_4h_aligned[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with volume confirmation
            if (close[i] > r1_4h_aligned[i] and volume_confirm[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 with volume confirmation
            elif (close[i] < s1_4h_aligned[i] and volume_confirm[i]):
                signals[i] = -0.20
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below S1
            if close[i] < s1_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:
            # Short: exit if price breaks above R1
            if close[i] > r1_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals