#!/usr/bin/env python3
name = "12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily Camarilla pivot levels (from previous day)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each daily bar
    # Pivot = (H + L + C) / 3
    pivot_d = (high_1d + low_1d + close_1d) / 3.0
    # Range = H - L
    range_d = high_1d - low_1d
    # R3 = C + (H - L) * 1.1 / 2
    r3_d = close_1d + range_d * 1.1 / 2.0
    # S3 = C - (H - L) * 1.1 / 2
    s3_d = close_1d - range_d * 1.1 / 2.0
    
    # Daily trend: close above/below pivot
    daily_trend_up = close_1d > pivot_d
    daily_trend_down = close_1d < pivot_d
    
    # Align daily data to 12h timeframe
    pivot_d_aligned = align_htf_to_ltf(prices, df_1d, pivot_d)
    r3_d_aligned = align_htf_to_ltf(prices, df_1d, r3_d)
    s3_d_aligned = align_htf_to_ltf(prices, df_1d, s3_d)
    daily_trend_up_aligned = align_htf_to_ltf(prices, df_1d, daily_trend_up)
    daily_trend_down_aligned = align_htf_to_ltf(prices, df_1d, daily_trend_down)
    
    # Volume spike detection: current volume > 2 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # ensure volume MA has enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(pivot_d_aligned[i]) or np.isnan(r3_d_aligned[i]) or np.isnan(s3_d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: daily trend up + price breaks above R3 + volume spike
            if (daily_trend_up_aligned[i] and 
                close[i] > r3_d_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: daily trend down + price breaks below S3 + volume spike
            elif (daily_trend_down_aligned[i] and 
                  close[i] < s3_d_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below pivot OR daily trend changes
            if close[i] < pivot_d_aligned[i] or not daily_trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above pivot OR daily trend changes
            if close[i] > pivot_d_aligned[i] or not daily_trend_down_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals