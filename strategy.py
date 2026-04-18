#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_Volume
Hypothesis: Use Camarilla pivot levels (R1/S1) from 1d as entry triggers with volume confirmation.
In bull markets, buy when price breaks above R1; in bear markets, sell when price breaks below S1.
Camarilla levels provide high-probability reversal/breakout points. Volume > 1.5x 20-period average
confirms institutional participation. This combination reduces false breakouts and captures strong moves.
Target: 15-25 trades/year by requiring both price level breach and volume confirmation.
Works in bull markets via R1 breakouts, bear markets via S1 breakdowns, and ranges via mean reversion at H4/L4.
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
    
    # Get 1d data for Camarilla pivot levels (HTF)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    # R4 = close + ((high - low) * 1.500)
    # R3 = close + ((high - low) * 1.250)
    # R2 = close + ((high - low) * 1.166)
    # R1 = close + ((high - low) * 1.083)
    # S1 = close - ((high - low) * 1.083)
    # S2 = close - ((high - low) * 1.166)
    # S3 = close - ((high - low) * 1.250)
    # S4 = close - ((high - low) * 1.500)
    range_1d = high_1d - low_1d
    r1 = close_1d + (range_1d * 1.083)
    s1 = close_1d - (range_1d * 1.083)
    
    # Align Camarilla levels to 12h timeframe (wait for bar close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
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
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above R1 with volume confirmation
            if close[i] > r1_aligned[i] and vol_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S1 with volume confirmation
            elif close[i] < s1_aligned[i] and vol_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price returns below S1 (mean reversion) or loses momentum
            if close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above R1 (mean reversion) or loses momentum
            if close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_Volume"
timeframe = "12h"
leverage = 1.0