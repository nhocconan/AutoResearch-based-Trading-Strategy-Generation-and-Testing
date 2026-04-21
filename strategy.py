#!/usr/bin/env python3
"""
12h_1d_Camarilla_R1S1_Breakout_Volume_Pullback
Hypothesis: Breakout pullback strategy using 1d Camarilla levels (R1/S1) on 12h timeframe.
Enter on retest of breakout level after initial breakout with volume confirmation.
Long: Price breaks above R1 (volume > 1.5x avg), then pulls back to touch R1.
Short: Price breaks below S1 (volume > 1.5x avg), then pulls back to touch S1.
Exit when price crosses the pivot point (PP).
Works in bull markets by buying breakout pullbacks and in bear markets by selling breakdown pullbacks.
Pullback entry reduces false breakouts and improves risk-reward.
Target: 15-30 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data once for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla pivot levels (based on previous day)
    pp = np.full_like(close_1d, np.nan)
    r1 = np.full_like(close_1d, np.nan)
    s1 = np.full_like(close_1d, np.nan)
    
    for i in range(1, len(high_1d)):
        pp[i] = (high_1d[i-1] + low_1d[i-1] + close_1d[i-1]) / 3.0
        r1[i] = close_1d[i-1] + (high_1d[i-1] - low_1d[i-1]) * 1.1 / 12.0
        s1[i] = close_1d[i-1] - (high_1d[i-1] - low_1d[i-1]) * 1.1 / 12.0
    
    # Shift to align with current day (levels are based on previous day)
    pp = np.roll(pp, 1)
    r1 = np.roll(r1, 1)
    s1 = np.roll(s1, 1)
    pp[0] = np.nan
    r1[0] = np.nan
    s1[0] = np.nan
    
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    breakout_state = 0  # 0: none, 1: long breakout, -1: short breakout
    
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
            # Check for new breakouts
            if price > r1_aligned[i] and volume_ok:
                breakout_state = 1  # Long breakout detected
            elif price < s1_aligned[i] and volume_ok:
                breakout_state = -1  # Short breakout detected
            
            # Enter on pullback to breakout level
            if breakout_state == 1 and abs(price - r1_aligned[i]) < 0.001 * r1_aligned[i]:
                signals[i] = 0.25
                position = 1
                breakout_state = 0  # Reset after entry
            elif breakout_state == -1 and abs(price - s1_aligned[i]) < 0.001 * s1_aligned[i]:
                signals[i] = -0.25
                position = -1
                breakout_state = 0  # Reset after entry
        
        elif position == 1:
            # Long exit: price crosses back below pivot point
            if price < pp_aligned[i]:
                signals[i] = 0.0
                position = 0
                breakout_state = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back above pivot point
            if price > pp_aligned[i]:
                signals[i] = 0.0
                position = 0
                breakout_state = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_Camarilla_R1S1_Breakout_Volume_Pullback"
timeframe = "12h"
leverage = 1.0