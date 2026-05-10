#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS
Hypothesis: Buy near daily Camarilla S1 in uptrend (price > 1d EMA34) and sell near daily R1 in downtrend (price < 1d EMA34) with volume confirmation. Uses 1d EMA34 for trend filter and Camarilla levels for precise entry/exit. Designed for 20-30 trades/year to minimize fee drag while capturing institutional reaction to key levels.
"""

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla levels (using previous day's range)
    # Camarilla formula: range = high - low
    # S1 = close - (range * 1.1 / 12)
    # S2 = close - (range * 1.1 / 6)
    # S3 = close - (range * 1.1 / 4)
    # R1 = close + (range * 1.1 / 12)
    # R2 = close + (range * 1.1 / 6)
    # R3 = close + (range * 1.1 / 4)
    range_1d = high_1d - low_1d
    s1_1d = close_1d - (range_1d * 1.1 / 12)
    s2_1d = close_1d - (range_1d * 1.1 / 6)
    s3_1d = close_1d - (range_1d * 1.1 / 4)
    r1_1d = close_1d + (range_1d * 1.1 / 12)
    r2_1d = close_1d + (range_1d * 1.1 / 6)
    r3_1d = close_1d + (range_1d * 1.1 / 4)
    
    # Align daily Camarilla levels to 4h timeframe (wait for daily bar close)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    
    # Daily EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 4h data for signal generation
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need daily data (34 for EMA) and ensure alignment
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or
            np.isnan(r1_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price vs daily EMA34
        uptrend_1d = close[i] > ema34_1d_aligned[i]
        downtrend_1d = close[i] < ema34_1d_aligned[i]
        
        # Volume filter: current 4h volume > 1.5x 20-period EMA
        vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
        volume_filter = volume[i] > vol_ema20[i] * 1.5
        
        # Distance to daily levels (as fraction of price)
        dist_to_s1 = (close[i] - s1_1d_aligned[i]) / close[i] if s1_1d_aligned[i] > 0 else 1.0
        dist_to_r1 = (r1_1d_aligned[i] - close[i]) / close[i] if r1_1d_aligned[i] > 0 else 1.0
        
        # Entry zones: within 0.3% of daily S1 or R1
        near_s1 = abs(dist_to_s1) < 0.003
        near_r1 = abs(dist_to_r1) < 0.003
        
        if position == 0:
            # Long entry: near S1 + uptrend + volume
            if near_s1 and uptrend_1d and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short entry: near R1 + downtrend + volume
            elif near_r1 and downtrend_1d and volume_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reaches S2 or trend fails
            if close[i] <= s2_1d_aligned[i] or not uptrend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reaches R2 or trend fails
            if close[i] >= r2_1d_aligned[i] or not downtrend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals