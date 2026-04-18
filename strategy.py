#!/usr/bin/env python3
"""
12h_1w_Camarilla_R1S1_Breakout_Volume
Hypothesis: Use weekly Camarilla R1/S1 levels as structural support/resistance on 12h timeframe. 
Breakouts above R1 or below S1 with volume confirmation (>1.5x 20-period average) capture 
institutional interest in trending markets. Weekly timeframe reduces noise and false breakouts, 
while 12h provides timely entries. Works in bull markets via R1 breakouts and bear markets via 
S1 breakdowns. Targets 15-25 trades/year by requiring alignment of weekly level breakout and 
volume confirmation.
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
    
    # Get weekly data for Camarilla levels (HTF)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla R1 and S1
    rng_1w = high_1w - low_1w
    r1_1w = close_1w + rng_1w * 1.1 / 12
    s1_1w = close_1w - rng_1w * 1.1 / 12
    
    # Align levels to 12h timeframe (wait for weekly bar close)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
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
        if (np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above weekly R1, with volume
            if close[i] > r1_1w_aligned[i] and vol_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below weekly S1, with volume
            elif close[i] < s1_1w_aligned[i] and vol_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price returns below weekly S1 (failed breakout/reversal)
            if close[i] < s1_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above weekly R1 (failed breakout/reversal)
            if close[i] > r1_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1w_Camarilla_R1S1_Breakout_Volume"
timeframe = "12h"
leverage = 1.0