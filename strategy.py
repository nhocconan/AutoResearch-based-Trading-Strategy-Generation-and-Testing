#!/usr/bin/env python3
"""
1d_WeeklyPivot_Breakout_Volume
Weekly pivot breakout strategy with volume confirmation for 1d timeframe:
- Long when price breaks above weekly R1 with volume > 1.5x average volume
- Short when price breaks below weekly S1 with volume > 1.5x average volume
- Exit when price returns to weekly pivot point (PP)
- Uses weekly pivot points calculated from prior week's high, low, close
- Designed for 15-25 trades/year per symbol
Works in both bull (captures breakouts) and bear (captures breakdowns) markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def calculate_weekly_pivot(high, low, close):
    """Calculate weekly pivot points: PP, R1, S1, R2, S2."""
    pp = (high + low + close) / 3
    r1 = 2 * pp - low
    s1 = 2 * pp - high
    r2 = pp + (high - low)
    s2 = pp - (high - low)
    return pp, r1, s1, r2, s2

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for pivot points
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) == 0:
        return np.zeros(n)
    
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly pivot points
    pp_weekly, r1_weekly, s1_weekly, r2_weekly, s2_weekly = calculate_weekly_pivot(
        high_weekly, low_weekly, close_weekly
    )
    
    # Align weekly pivots to daily timeframe
    pp_aligned = align_ltf_to_htf(prices, df_weekly, pp_weekly)
    r1_aligned = align_ltf_to_htf(prices, df_weekly, r1_weekly)
    s1_aligned = align_ltf_to_htf(prices, df_weekly, s1_weekly)
    
    # Calculate average volume (20-day)
    volume = prices['volume'].values
    vol_avg = np.full(n, np.nan)
    for i in range(19, n):
        vol_avg[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need volume average
    
    for i in range(start_idx, n):
        # Skip if pivot data not available
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.5x average volume
        volume_condition = volume[i] > 1.5 * vol_avg[i]
        
        if position == 0:
            # Long: break above R1 with volume confirmation
            if prices['close'][i] > r1_aligned[i] and volume_condition:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume confirmation
            elif prices['close'][i] < s1_aligned[i] and volume_condition:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: return to pivot point
            if prices['close'][i] <= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: return to pivot point
            if prices['close'][i] >= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyPivot_Breakout_Volume"
timeframe = "1d"
leverage = 1.0