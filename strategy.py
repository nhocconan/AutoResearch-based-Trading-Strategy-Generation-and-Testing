#!/usr/bin/env python3
"""
#100888 - 12h_Camarilla_R1_S1_Breakout_1wTrend_Volume
Hypothesis: Breakout at weekly pivot R1/S1 levels with volume confirmation and 1w EMA50 trend filter on 12h timeframe. Targets 12-37 trades/year to minimize fee drag. Uses weekly trend filter to capture major moves while avoiding counter-trend trades. Works in bull (breakouts with trend) and bear (mean reversion to pivot during range-bound periods).
"""

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
    
    # Get 1w data for EMA50 trend filter and weekly pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 1w EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate weekly pivot points from previous week (to avoid look-ahead)
    weekly_pivot = (high_1w + low_1w + close_1w) / 3
    weekly_range = high_1w - low_1w
    weekly_r1 = weekly_pivot + weekly_range * 1.0 / 2
    weekly_s1 = weekly_pivot - weekly_range * 1.0 / 2
    
    # Align to 12h timeframe (previous week's levels for current period)
    pivot_r1 = align_htf_to_ltf(prices, df_1w, weekly_r1)
    pivot_s1 = align_htf_to_ltf(prices, df_1w, weekly_s1)
    pivot_point = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Volume filter: volume > 1.8x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(pivot_r1[i]) or 
            np.isnan(pivot_s1[i]) or np.isnan(pivot_point[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: price breaks above R1, above 1w EMA50, volume spike
        if (close[i] > pivot_r1[i] and 
            close[i] > ema50_1w_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short condition: price breaks below S1, below 1w EMA50, volume spike
        elif (close[i] < pivot_s1[i] and 
              close[i] < ema50_1w_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: price returns to weekly pivot (mean reversion)
        elif position == 1 and close[i] < pivot_point[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > pivot_point[i]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0