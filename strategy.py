#!/usr/bin/env python3
# 6h_PivotRange_Reversal_12hTrend
# Hypothesis: Identifies reversal opportunities at daily pivot range boundaries with 12h trend alignment.
# Uses daily pivot range (S1-R1) as dynamic support/resistance. Enters long near S1 in 12h uptrend, short near R1 in 12h downtrend.
# Volume confirmation filters for institutional interest. Designed for 6h timeframe with 50-150 total trades over 4 years.
# Works in bull/bear by aligning with 12h trend - only takes trades in direction of higher timeframe momentum.
# Position size 0.25 for balanced risk management.

name = "6h_PivotRange_Reversal_12hTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate daily pivot points (standard formula)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    
    # Align pivot levels to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Calculate 12h EMA for trend filter
    ema_20_12h = pd.Series(df_12h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # Volume confirmation (20-period MA)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Warmup for EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_20_12h_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # 12h trend filter
        uptrend_12h = close[i] > ema_20_12h_aligned[i]
        downtrend_12h = close[i] < ema_20_12h_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        # Distance to pivot levels (for entry precision)
        dist_to_s1 = abs(close[i] - s1_aligned[i]) / s1_aligned[i]
        dist_to_r1 = abs(close[i] - r1_aligned[i]) / r1_aligned[i]
        near_s1 = dist_to_s1 < 0.005  # Within 0.5% of S1
        near_r1 = dist_to_r1 < 0.005  # Within 0.5% of R1
        
        if position == 0:
            # Long entry: near S1 in 12h uptrend with volume confirmation
            if near_s1 and uptrend_12h and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: near R1 in 12h downtrend with volume confirmation
            elif near_r1 and downtrend_12h and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reaches midpoint or trend changes
            midpoint = (s1_aligned[i] + r1_aligned[i]) / 2
            if close[i] >= midpoint or not uptrend_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reaches midpoint or trend changes
            midpoint = (s1_aligned[i] + r1_aligned[i]) / 2
            if close[i] <= midpoint or not downtrend_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals