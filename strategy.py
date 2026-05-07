#!/usr/bin/env python3
# 4H_Camarilla_Pivot_R3_S3_Breakout_1DTrend_Volume_Spike_v2
# Hypothesis: Refines the original strategy by adding a volume surge filter (volume > 2.5x 20-period average) and tightening the exit condition (price must close back below/above the pivot level). This reduces whipsaw and improves trade quality. Works in both bull and bear markets by only taking long breaks above R3 in uptrend (close > EMA34) and short breaks below S3 in downtrend (close < EMA34). Target: 15-30 trades per year with size 0.25 to minimize fee drag.

name = "4H_Camarilla_Pivot_R3_S3_Breakout_1DTrend_Volume_Spike_v2"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily OHLC for Camarilla pivots
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R3 = close + (high - low) * 1.1/4, S3 = close - (high - low) * 1.1/4
    rng = high_1d - low_1d
    camarilla_r3 = close_1d + rng * 1.1 / 4
    camarilla_s3 = close_1d - rng * 1.1 / 4
    
    # 1-day EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align Camarilla levels and EMA to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: current volume > 2.5x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure we have volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: surge confirmation (more stringent)
        volume_filter = volume[i] > 2.5 * vol_ma[i]
        
        if position == 0:
            # Long: Price breaks above R3 + Uptrend (price > EMA34) + volume surge
            if (close[i] > r3_aligned[i] and 
                close[i] > ema34_aligned[i] and
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 + Downtrend (price < EMA34) + volume surge
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema34_aligned[i] and
                  volume_filter):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price closes back below R3 (more strict than before)
            if close[i] < r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price closes back above S3 (more strict than before)
            if close[i] > s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals