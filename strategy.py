#!/usr/bin/env python3
"""
6H_Weekly_Pivot_Trend_Confirmation_v2
Hypothesis: Use weekly pivot levels (R1/S1) as structural support/resistance on 6h chart.
Long when price crosses above 6h EMA(20) and touches weekly R1 with volume confirmation.
Short when price crosses below 6h EMA(20) and touches weekly S1 with volume confirmation.
Weekly pivots provide multi-week context that adapts to both bull and bear markets,
while EMA(20) filters for intermediate trend and volume confirms conviction.
Designed for ~20-40 trades/year to minimize fee drag on 6h timeframe.
"""
name = "6H_Weekly_Pivot_Trend_Confirmation_v2"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for EMA calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Calculate 6h EMA(20) for trend filter
    close_6h = pd.Series(df_6h['close'])
    ema_6h = close_6h.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_6h_aligned = align_htf_to_ltf(prices, df_6h, ema_6h)
    
    # Get weekly data for pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points (R1, S1)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    r1_1w = pivot_1w + (range_1w * 1.1 / 12)
    s1_1w = pivot_1w - (range_1w * 1.1 / 12)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # Volume filter: current volume > 1.3 * 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    start_idx = max(20, 20)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(ema_6h_aligned[i]) or np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 24 bars between trades (4 days on 6h TF) to reduce frequency
            if bars_since_exit < 24:
                continue
                
            # Long: price crosses above EMA(20) and touches weekly R1 with volume
            if (close[i] > ema_6h_aligned[i] and close[i-1] <= ema_6h_aligned[i-1] and 
                low[i] <= r1_1w_aligned[i] and volume_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_exit = 0
            # Short: price crosses below EMA(20) and touches weekly S1 with volume
            elif (close[i] < ema_6h_aligned[i] and close[i-1] >= ema_6h_aligned[i-1] and 
                  high[i] >= s1_1w_aligned[i] and volume_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit: price returns to opposite EMA side
            if position == 1 and close[i] < ema_6h_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            elif position == -1 and close[i] > ema_6h_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals