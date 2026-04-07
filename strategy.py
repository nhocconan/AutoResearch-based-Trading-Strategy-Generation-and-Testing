#!/usr/bin/env python3
"""
6h_weekly_pivot_volume_v1
Hypothesis: Use weekly pivot points from weekly timeframe to determine trend direction, with 60-period volume moving average confirmation on 6h timeframe. Enter long when price breaks above weekly R1 with volume confirmation, short when price breaks below weekly S1. Exit on opposite pivot level breach. Designed for 50-150 total trades over 4 years (~12-37/year) to minimize fee drag while capturing trending moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly pivot points (using 1w timeframe)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivots: P = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    
    # Align weekly pivots to 6h timeframe (shifted by 1 for completed weekly bars)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Volume filter: 60-period average on 6h timeframe
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=60, min_periods=60).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if data not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation: require volume above average
        vol_ok = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below weekly pivot
            if close[i] < pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above weekly pivot
            if close[i] > pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Breakout above R1: go long
                if close[i] > r1_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Breakdown below S1: go short
                elif close[i] < s1_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals