#!/usr/bin/env python3
# 4h_1d_Camarilla_R1S1_Breakout_VolumeTrend
# Hypothesis: On 4h timeframe, trade breakouts from 1d-derived Camarilla R1/S1 levels with volume spike confirmation and 1d EMA trend filter.
# R1/S1 provide tighter breakout levels than R4/S4, improving win rate while 1d EMA34 filter ensures alignment with daily trend.
# Volume spike (2x 20-period average) confirms institutional participation. Designed for 20-40 trades/year to avoid fee drag.
# Works in bull markets (buy R1 breakouts in uptrends) and bear markets (sell S1 breakdowns in downtrends).

name = "4h_1d_Camarilla_R1S1_Breakout_VolumeTrend"
timeframe = "4h"
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
    
    # Calculate 1d R1 and S1 levels using previous day's data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point and range
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels: R1 and S1 (inner breakout levels)
    s1_1d = close_1d - (range_1d * 1.1 / 6)
    r1_1d = close_1d + (range_1d * 1.1 / 6)
    
    # 1d EMA34 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d levels to 4h timeframe
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume average for spike detection (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(ema_34_aligned[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above R1, volume spike, and price above 1d EMA34 (uptrend)
            if (close[i] > r1_aligned[i] * 1.002 and 
                volume[i] > 2.0 * volume_ma[i] and
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below S1, volume spike, and price below 1d EMA34 (downtrend)
            elif (close[i] < s1_aligned[i] * 0.998 and 
                  volume[i] > 2.0 * volume_ma[i] and
                  close[i] < ema_34_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price below S1 or trend reversal (below EMA34)
            if close[i] < s1_aligned[i] * 0.998 or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above R1 or trend reversal (above EMA34)
            if close[i] > r1_aligned[i] * 1.002 or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals