#!/usr/bin/env python3
"""
12h_Pivot_R1_S1_1dTrend_Volume
Hypothesis: Daily pivot-based resistance (R1) and support (S1) levels act as
significant barriers where breakouts with volume and trend alignment signal
institutional participation. Using 12h timeframe reduces trade frequency to
avoid fee drag while capturing multi-day moves. Works in bull markets via
long breakouts and in bear markets via short breakdowns.
"""

name = "12h_Pivot_R1_S1_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data for pivot levels, trend, and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily pivot point and support/resistance levels
    # Pivot = (High + Low + Close) / 3
    # R1 = (2 * Pivot) - Low
    # S1 = (2 * Pivot) - High
    pivot = (high_1d + low_1d + close_1d) / 3.0
    pivot_r1 = (2 * pivot) - low_1d
    pivot_s1 = (2 * pivot) - high_1d
    
    # Align pivot levels to 12h (based on prior day's data)
    pivot_r1_aligned = align_htf_to_ltf(prices, df_1d, pivot_r1)
    pivot_s1_aligned = align_htf_to_ltf(prices, df_1d, pivot_s1)
    
    # Daily trend filter: EMA34
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Daily volume average for spike detection (20-day average)
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # 12h price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA34 (34) and volume average (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(pivot_r1_aligned[i]) or 
            np.isnan(pivot_s1_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_avg_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Daily trend filter
        uptrend_1d = close[i] > ema34_1d_aligned[i]
        downtrend_1d = close[i] < ema34_1d_aligned[i]
        
        # Volume filter: current 12h volume > 2.0x average daily volume (scaled)
        # 1 day = 2 x 12h bars, so scale daily volume to 12h equivalent
        vol_12h_equiv = vol_avg_1d_aligned[i] / 2.0
        volume_spike = volume[i] > vol_12h_equiv * 2.0
        
        if position == 0:
            # Long entry: price breaks above R1 + uptrend + volume spike
            if high[i] > pivot_r1_aligned[i] and uptrend_1d and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S1 + downtrend + volume spike
            elif low[i] < pivot_s1_aligned[i] and downtrend_1d and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below S1 or trend fails
            if low[i] < pivot_s1_aligned[i] or not uptrend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above R1 or trend fails
            if high[i] > pivot_r1_aligned[i] or not downtrend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals