#!/usr/bin/env python3
"""
6h Weekly Pivot Breakout with Volume Confirmation
Hypothesis: Weekly pivot levels (especially S3/R3) act as strong support/resistance in both bull and bear markets. 
Breakouts above R3 or below S3 with volume confirmation capture institutional participation. 
This strategy targets 6h timeframe to avoid excessive trading while capturing meaningful moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def calculate_weekly_pivot(high, low, close):
    """Calculate weekly pivot points and support/resistance levels"""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    return pivot, r1, r2, r3, s1, s2, s3

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivots
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate pivots for each week
    pivots = np.zeros_like(weekly_close)
    r1_vals = np.zeros_like(weekly_close)
    r2_vals = np.zeros_like(weekly_close)
    r3_vals = np.zeros_like(weekly_close)
    s1_vals = np.zeros_like(weekly_close)
    s2_vals = np.zeros_like(weekly_close)
    s3_vals = np.zeros_like(weekly_close)
    
    for i in range(len(weekly_close)):
        p, r1, r2, r3, s1, s2, s3 = calculate_weekly_pivot(
            weekly_high[i], weekly_low[i], weekly_close[i]
        )
        pivots[i] = p
        r1_vals[i] = r1
        r2_vals[i] = r2
        r3_vals[i] = r3
        s1_vals[i] = s1
        s2_vals[i] = s2
        s3_vals[i] = s3
    
    # Align weekly pivots to 6h timeframe (wait for weekly close)
    pivots_aligned = align_htf_to_ltf(prices, df_weekly, pivots)
    r3_aligned = align_htf_to_ltf(prices, df_weekly, r3_vals)
    s3_aligned = align_htf_to_ltf(prices, df_weekly, s3_vals)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[max(0, i-19):i+1])
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    vol_threshold = vol_ma * 1.5
    vol_ok = volume > vol_threshold
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Warmup
    
    for i in range(start_idx, n):
        if np.isnan(pivots_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]):
            signals[i] = 0.0
            continue
        
        pivot_val = pivots_aligned[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        vol_ok_current = vol_ok[i]
        
        if position == 0:
            # Enter long: break above R3 with volume
            if close[i] > r3_val and vol_ok_current:
                signals[i] = 0.25
                position = 1
            # Enter short: break below S3 with volume
            elif close[i] < s3_val and vol_ok_current:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below pivot or weekly close below pivot
            if close[i] < pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above pivot or weekly close above pivot
            if close[i] > pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_Breakout_Volume"
timeframe = "6h"
leverage = 1.0