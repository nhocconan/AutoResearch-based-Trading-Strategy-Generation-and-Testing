#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_Volume
Hypothesis: Enter long when price breaks above daily Camarilla R1 level and short when price breaks below S1 level, with trend filter from 1d EMA34 and volume confirmation. Exit when price crosses the daily pivot point. Uses 12h timeframe to reduce trade frequency and avoid fee drag. Designed to work in both bull and bear markets by following the daily trend.
"""

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
    
    # Get 1d data for trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla pivot levels for each day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    # Pivot = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    pivot_1d = (high_1d + low_1d + close_1d_arr) / 3.0
    r1_1d = close_1d_arr + (high_1d - low_1d) * 1.1 / 12.0
    s1_1d = close_1d_arr - (high_1d - low_1d) * 1.1 / 12.0
    
    # Align to 12h timeframe (previous day's levels available at open)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Volume filter: require volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup period
    start_idx = 34  # need 34 for EMA34
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 in uptrend with volume confirmation
            if (close[i] > r1_aligned[i] and 
                close[i] > ema34_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 in downtrend with volume confirmation
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price crosses below pivot point
            if close[i] < pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above pivot point
            if close[i] > pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0