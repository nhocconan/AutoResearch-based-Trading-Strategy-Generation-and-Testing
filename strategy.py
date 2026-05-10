#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_Breakout_1dTrend_Volume
Hypothesis: Uses Camarilla pivot levels (S1, R1) from daily timeframe for breakout entries,
with 1d EMA trend filter and volume confirmation. Designed to work in both bull and bear
markets by capturing momentum in the direction of the daily trend. Target: 15-30 trades/year.
"""

name = "12h_Camarilla_Pivot_Breakout_1dTrend_Volume"
timeframe = "12h"
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
    
    # Convert to Series for indicator calculations
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    volume_s = pd.Series(volume)
    
    # Calculate Camarilla pivot levels from daily data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Use previous day's OHLC for today's Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot and Camarilla levels
    pivot = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    
    # Camarilla levels: S1, S2, S3, S4, R1, R2, R3, R4
    # S1 = close - (range * 1.0833)
    # R1 = close + (range * 1.0833)
    s1 = close_1d - (range_hl * 1.0833)
    r1 = close_1d + (range_hl * 1.0833)
    
    # Calculate 1d EMA trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1d_up = close_1d > ema34_1d
    trend_1d_down = close_1d < ema34_1d
    
    # Align 1d indicators to 12h timeframe
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down.astype(float))
    
    # Volume confirmation (20-period average)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data for all indicators
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]) or
            np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        if position == 0:
            # Long entry: price breaks above R1 with uptrend and volume
            if (close[i] > r1_aligned[i] and 
                trend_1d_up_aligned[i] > 0.5 and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S1 with downtrend and volume
            elif (close[i] < s1_aligned[i] and 
                  trend_1d_down_aligned[i] > 0.5 and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S1 or trend changes
            if (close[i] < s1_aligned[i] or trend_1d_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above R1 or trend changes
            if (close[i] > r1_aligned[i] or trend_1d_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals