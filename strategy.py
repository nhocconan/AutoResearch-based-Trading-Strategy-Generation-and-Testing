#!/usr/bin/env python3
"""
4h_12h_Camarilla_R1S1_Breakout_Volume
Hypothesis: Use Camarilla pivot levels (R1, S1) from 12h timeframe with volume confirmation on 4h.
Long when price breaks above R1 with volume > 1.5x average volume.
Short when price breaks below S1 with volume > 1.5x average volume.
Exit when price crosses back through the pivot point (PP).
Designed for 4h timeframe to capture multi-day moves with ~20-50 trades/year.
Works in bull markets by buying breakouts and in bear markets by selling breakdowns.
Volume confirmation filters false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data once for Camarilla pivots
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Camarilla pivot levels (based on previous 12h bar)
    # PP = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    pp = np.full_like(close_12h, np.nan)
    r1 = np.full_like(close_12h, np.nan)
    s1 = np.full_like(close_12h, np.nan)
    
    for i in range(1, len(high_12h)):
        pp[i] = (high_12h[i-1] + low_12h[i-1] + close_12h[i-1]) / 3.0
        r1[i] = close_12h[i-1] + (high_12h[i-1] - low_12h[i-1]) * 1.1 / 12.0
        s1[i] = close_12h[i-1] - (high_12h[i-1] - low_12h[i-1]) * 1.1 / 12.0
    
    # Shift to align with current 12h bar (levels are based on previous 12h bar)
    pp = np.roll(pp, 1)
    r1 = np.roll(r1, 1)
    s1 = np.roll(s1, 1)
    pp[0] = np.nan
    r1[0] = np.nan
    s1[0] = np.nan
    
    pp_aligned = align_htf_to_ltf(prices, df_12h, pp)
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.5 * vol_ma
        else:
            volume_ok = False
        
        if position == 0:
            # Long conditions: break above R1 + volume confirmation
            if price > r1_aligned[i] and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short conditions: break below S1 + volume confirmation
            elif price < s1_aligned[i] and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back below pivot point
            if price < pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back above pivot point
            if price > pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_12h_Camarilla_R1S1_Breakout_Volume"
timeframe = "4h"
leverage = 1.0