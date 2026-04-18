#!/usr/bin/env python3
"""
12h_Pivot_R1S1_Breakout_Volume_Regime
Pivot-based breakout strategy for 12h timeframe with volume confirmation and regime filter:
- Long when price breaks above R1 with volume confirmation and favorable regime
- Short when price breaks below S1 with volume confirmation and favorable regime
- Uses 1d Pivot points (R1, S1) as key levels
- Volume filter: current volume > 1.5x average volume
- Regime filter: Choppiness Index < 61.8 (trending market)
- Designed for 12-37 trades/year per symbol
Works in both bull (captures breakouts) and bear (breakdowns) markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_pivot_points(high, low, close):
    """Calculate standard pivot points and support/resistance levels."""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    return pivot, r1, s1, r2, s2

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index to identify trending vs ranging markets."""
    n = len(high)
    atr = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(n):
        if i == 0:
            tr[i] = high[i] - low[i]
        else:
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Calculate ATR using Wilder's smoothing
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    # Calculate sum of true range over period
    tr_sum = np.zeros(n)
    for i in range(period-1, n):
        tr_sum[i] = np.sum(tr[i-period+1:i+1])
    
    # Calculate Choppiness Index
    chop = np.full(n, 50.0)  # default to middle value
    for i in range(period-1, n):
        if tr_sum[i] > 0:
            chop[i] = 100 * np.log10(tr_sum[i] / (atr[i] * period)) / np.log10(period)
    
    return chop

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Pivot points
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Pivot points
    _, r1_1d, s1_1d, _, _ = calculate_pivot_points(high_1d, low_1d, close_1d)
    
    # Align 1d Pivot points to 12h timeframe
    r1_1d_12h = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_12h = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Calculate average volume (20-period)
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Calculate Choppiness Index for regime filter
    chop = calculate_choppiness(high, low, close)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # need sufficient data for calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_1d_12h[i]) or np.isnan(s1_1d_12h[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x average volume
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        # Regime filter: Choppiness Index < 61.8 (trending market)
        regime_filter = chop[i] < 61.8
        
        if position == 0:
            # Long: price breaks above R1 with volume and regime confirmation
            if close[i] > r1_1d_12h[i] and volume_filter and regime_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume and regime confirmation
            elif close[i] < s1_1d_12h[i] and volume_filter and regime_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S1 or regime becomes unfavorable
            if close[i] < s1_1d_12h[i] or chop[i] >= 61.8:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1 or regime becomes unfavorable
            if close[i] > r1_1d_12h[i] or chop[i] >= 61.8:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Pivot_R1S1_Breakout_Volume_Regime"
timeframe = "12h"
leverage = 1.0