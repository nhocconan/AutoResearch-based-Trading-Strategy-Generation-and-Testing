#!/usr/bin/env python3
"""
4H_Camarilla_R1_S1_Breakout_1DTrend_Filter
Hypothesis: On 4h timeframe, buy when price breaks above Camarilla R1 level and sell/short when price breaks below S1 level, filtered by 1d EMA trend to avoid counter-trend trades. Uses volume spike confirmation to ensure breakout validity. This structure provides low-frequency, high-conviction trades suitable for both bull and bear markets by aligning with higher timeframe trend.
"""

name = "4H_Camarilla_R1_S1_Breakout_1DTrend_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 4h data for Camarilla calculation
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels using previous day's OHLC
    # For each 4h bar, we need the previous day's high, low, close
    # We'll use the 1d data shifted by 1 to get previous day's values
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(df_1d['high'].values, 1)
    prev_low_1d = np.roll(df_1d['low'].values, 1)
    
    # Align previous day's values to 4h timeframe
    prev_close_1d_aligned = align_htf_to_ltf(prices, df_1d, prev_close_1d)
    prev_high_1d_aligned = align_htf_to_ltf(prices, df_1d, prev_high_1d)
    prev_low_1d_aligned = align_htf_to_ltf(prices, df_1d, prev_low_1d)
    
    # Calculate Camarilla R1 and S1 for each 4h bar
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    camarilla_range = prev_high_1d_aligned - prev_low_1d_aligned
    r1 = prev_close_1d_aligned + 1.1 * camarilla_range / 12
    s1 = prev_close_1d_aligned - 1.1 * camarilla_range / 12
    
    # Volume filter: current 4h volume > 1.5x 20-period EMA of volume
    volume_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > volume_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for EMA and volume EMA
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(r1[i]) or 
            np.isnan(s1[i]) or
            np.isnan(volume_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price vs 1d EMA50
        uptrend_1d = close[i] > ema50_1d_aligned[i]
        downtrend_1d = close[i] < ema50_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 in uptrend with volume
            if close[i] > r1[i] and uptrend_1d and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 in downtrend with volume
            elif close[i] < s1[i] and downtrend_1d and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S1 or trend fails
            if close[i] < s1[i] or not uptrend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R1 or trend fails
            if close[i] > r1[i] or not downtrend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals