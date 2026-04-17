#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe with 1w/1d structure.
Use weekly pivot points (PP) from 1w to determine trend direction (above/below PP).
Enter long when price crosses above 1d R1 with volume confirmation and price > 1w PP.
Enter short when price crosses below 1d S1 with volume confirmation and price < 1w PP.
Exit on opposite pivot level (S1 for long, R1 for short) or when price crosses 1w PP in opposite direction.
Uses weekly structure for trend and daily pivots for precise entries to keep trade frequency low (12-30/year).
Works in bull markets via trend-following breaks above weekly PP and in bear via mean-reversion at daily S1/R1.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf  # Note: corrected import name

def calculate_pivot_points(high, low, close):
    """Calculate classic pivot points: P, R1, R2, S1, S2"""
    pp = (high + low + close) / 3.0
    r1 = 2 * pp - low
    s1 = 2 * pp - high
    r2 = pp + (high - low)
    s2 = pp - (high - low)
    return pp, r1, r2, s1, s2

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend (pivot points)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points
    pp_1w, r1_1w, r2_1w, s1_1w, s2_1w = calculate_pivot_points(high_1w, low_1w, close_1w)
    
    # Get daily data for entry levels (pivot points)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points
    pp_1d, r1_1d, r2_1d, s1_1d, s2_1d = calculate_pivot_points(high_1d, low_1d, close_1d)
    
    # Align weekly and daily data to 6h
    pp_1w_aligned = align_ltf_to_htf(prices, df_1w, pp_1w)
    r1_1d_aligned = align_ltf_to_htf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_ltf_to_htf(prices, df_1d, s1_1d)
    pp_1d_aligned = align_ltf_to_htf(prices, df_1d, pp_1d)  # for exit condition
    
    # Volume filter: current volume > 1.5x 24-period average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 30  # Need enough data for weekly/daily calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pp_1w_aligned[i]) or np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or np.isnan(pp_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price crosses above daily R1 with volume and above weekly PP
            if close[i] > r1_1d_aligned[i] and close[i-1] <= r1_1d_aligned[i-1] and volume_filter[i] and close[i] > pp_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below daily S1 with volume and below weekly PP
            elif close[i] < s1_1d_aligned[i] and close[i-1] >= s1_1d_aligned[i-1] and volume_filter[i] and close[i] < pp_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below daily S1 or crosses below weekly PP
            if close[i] < s1_1d_aligned[i] and close[i-1] >= s1_1d_aligned[i-1] or close[i] < pp_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above daily R1 or crosses above weekly PP
            if close[i] > r1_1d_aligned[i] and close[i-1] <= r1_1d_aligned[i-1] or close[i] > pp_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1wPP_1dR1S1_Volume"
timeframe = "6h"
leverage = 1.0