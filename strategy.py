#!/usr/bin/env python3
"""
1d_Weekly_Pivot_R1_S1_Breakout_Volume
Hypothesis: Uses weekly Camarilla pivot levels (R1, S1) for breakout signals on daily timeframe.
Combines price level breakouts with volume confirmation to reduce false signals.
Works in both bull and bear markets by capturing breakouts from key weekly support/resistance levels.
Target: 15-25 trades/year to minimize fee drag while maintaining edge.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close"""
    range_val = high - low
    if range_val == 0:
        return close, close, close, close
    c = close
    r1 = c + (range_val * 1.1 / 12)
    s1 = c - (range_val * 1.1 / 12)
    return r1, s1

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla pivot levels
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly Camarilla levels (R1, S1)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    r1_weekly = np.full(len(weekly_high), np.nan)
    s1_weekly = np.full(len(weekly_low), np.nan)
    
    for i in range(len(weekly_high)):
        r1, s1 = calculate_camarilla(weekly_high[i], weekly_low[i], weekly_close[i])
        r1_weekly[i] = r1
        s1_weekly[i] = s1
    
    # Volume spike: current volume > 1.5 x 20-day average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    # Align weekly levels to daily timeframe
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1_weekly)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1_weekly)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above weekly R1 with volume spike
            if close[i] > r1_aligned[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below weekly S1 with volume spike
            elif close[i] < s1_aligned[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns below weekly R1
            if close[i] < r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above weekly S1
            if close[i] > s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Weekly_Pivot_R1_S1_Breakout_Volume"
timeframe = "1d"
leverage = 1.0