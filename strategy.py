#!/usr/bin/env python3
# 6h_1d_Pivot_R4S4_Breakout_VolumeTrend
# Hypothesis: On 6h timeframe, trade breakouts from 1d-derived R4/S4 levels with volume spike confirmation and 1d EMA trend filter.
# R4/S4 represent stronger breakout levels than R3/S3, reducing false signals. Uses 1d EMA34 to filter trades in trending markets.
# Targets 15-30 trades per year by requiring strong breakouts with volume confirmation.
# Works in both bull and bear markets by aligning with 1d trend direction.

name = "6h_1d_Pivot_R4S4_Breakout_VolumeTrend"
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
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d R4 and S4 levels using previous day's data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point and range
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels: R4 and S4 (stronger breakout levels)
    s4_1d = close_1d - (range_1d * 1.1 / 2)
    r4_1d = close_1d + (range_1d * 1.1 / 2)
    
    # 1d EMA34 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d levels to 6h timeframe
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume average for spike detection (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(s4_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(ema_34_aligned[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above R4, volume spike, and price above 1d EMA34 (uptrend)
            if (close[i] > r4_aligned[i] * 1.003 and 
                volume[i] > 2.0 * volume_ma[i] and
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below S4, volume spike, and price below 1d EMA34 (downtrend)
            elif (close[i] < s4_aligned[i] * 0.997 and 
                  volume[i] > 2.0 * volume_ma[i] and
                  close[i] < ema_34_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price below S4 or trend reversal (below EMA34)
            if close[i] < s4_aligned[i] * 0.997 or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above R4 or trend reversal (above EMA34)
            if close[i] > r4_aligned[i] * 1.003 or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals