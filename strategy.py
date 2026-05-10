#!/usr/bin/env python3
# 6h_Weekly_Pivot_Reversal_With_Volume
# Hypothesis: Weekly pivot levels act as strong support/resistance in crypto markets.
# Price often reverses when approaching weekly S1/R1 or breaks through S2/R2 with momentum.
# We use weekly pivot points (calculated from prior week) and enter on rejection of S1/R1
# or breakout of S2/R2 with volume confirmation. Works in both trending and ranging markets.
# Weekly timeframe provides structural context while 6h entries capture shorter-term moves.

name = "6h_Weekly_Pivot_Reversal_With_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def calculate_weekly_pivots(high, low, close):
    """Calculate weekly pivot points: P, R1, S1, R2, S2"""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    return pivot, r1, s1, r2, s2

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
    if len(df_weekly) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points from prior week's data
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate pivots for each week
    pivots = np.full_like(weekly_close, np.nan)
    r1_vals = np.full_like(weekly_close, np.nan)
    s1_vals = np.full_like(weekly_close, np.nan)
    r2_vals = np.full_like(weekly_close, np.nan)
    s2_vals = np.full_like(weekly_close, np.nan)
    
    for i in range(1, len(weekly_close)):  # Start from 1 to use prior week
        pivot, r1, s1, r2, s2 = calculate_weekly_pivots(
            weekly_high[i-1], weekly_low[i-1], weekly_close[i-1]
        )
        pivots[i] = pivot
        r1_vals[i] = r1
        s1_vals[i] = s1
        r2_vals[i] = r2
        s2_vals[i] = s2
    
    # Align weekly pivots to 6h timeframe (with 1-week delay for prior week data)
    pivots_aligned = align_ltf_to_htf(prices, df_weekly, pivots)
    r1_aligned = align_ltf_to_htf(prices, df_weekly, r1_vals)
    s1_aligned = align_ltf_to_htf(prices, df_weekly, s1_vals)
    r2_aligned = align_ltf_to_htf(prices, df_weekly, r2_vals)
    s2_aligned = align_ltf_to_htf(prices, df_weekly, s2_vals)
    
    # Volume confirmation (20-period MA on 6h chart)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly data alignment and volume MA
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(pivots_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(r2_aligned[i]) or 
            np.isnan(s2_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Price action relative to weekly pivot levels
        at_s1 = abs(close[i] - s1_aligned[i]) / s1_aligned[i] < 0.005  # Within 0.5% of S1
        at_r1 = abs(close[i] - r1_aligned[i]) / r1_aligned[i] < 0.005  # Within 0.5% of R1
        above_r2 = close[i] > r2_aligned[i]  # Above R2
        below_s2 = close[i] < s2_aligned[i]  # Below S2
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: rejection of S1 (bounce) or breakout of R2 with volume
            if (at_s1 and close[i] > close[i-1]) or (above_r2 and volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short entry: rejection of R1 (reversal) or breakdown of S2 with volume
            elif (at_r1 and close[i] < close[i-1]) or (below_s2 and volume_confirm):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: rejection of R1 or breakdown below pivot
            if at_r1 and close[i] < close[i-1]:
                signals[i] = 0.0
                position = 0
            elif close[i] < pivots_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: rejection of S1 or breakout above pivot
            if at_s1 and close[i] > close[i-1]:
                signals[i] = 0.0
                position = 0
            elif close[i] > pivots_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals