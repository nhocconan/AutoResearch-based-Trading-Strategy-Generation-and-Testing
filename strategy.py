#!/usr/bin/env python3
# 6h_1w_Pivot_R4S4_Breakout_VolumeTrend
# Hypothesis: On 6h timeframe, trade breakouts from weekly-derived R4/S4 levels with volume spike confirmation and 1d EMA trend filter.
# R4/S4 from weekly pivot represent stronger breakout levels, reducing false signals. Uses 1d EMA34 to filter trades in trending markets.
# Targets 15-30 trades per year by requiring strong breakouts with volume confirmation.
# Works in both bull and bear markets by aligning with 1d trend direction.

name = "6h_1w_Pivot_R4S4_Breakout_VolumeTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate weekly R4 and S4 levels using previous week's data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Pivot point and range
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    
    # Camarilla levels: R4 and S4 (stronger breakout levels)
    s4_1w = close_1w - (range_1w * 1.1 / 2)
    r4_1w = close_1w + (range_1w * 1.1 / 2)
    
    # 1d EMA34 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly levels to 6h timeframe
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    
    # Align 1d EMA to 6h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume average for spike detection (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(s4_1w_aligned[i]) or np.isnan(r4_1w_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(ema_34_1d_aligned[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above weekly R4, volume spike, and price above 1d EMA34 (uptrend)
            if (close[i] > r4_1w_aligned[i] * 1.003 and 
                volume[i] > 2.0 * volume_ma[i] and
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below weekly S4, volume spike, and price below 1d EMA34 (downtrend)
            elif (close[i] < s4_1w_aligned[i] * 0.997 and 
                  volume[i] > 2.0 * volume_ma[i] and
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price below weekly S4 or trend reversal (below EMA34)
            if close[i] < s4_1w_aligned[i] * 0.997 or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above weekly R4 or trend reversal (above EMA34)
            if close[i] > r4_1w_aligned[i] * 1.003 or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals