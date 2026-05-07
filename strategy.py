#/usr/bin/env python3
"""
12h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_HT
Hypothesis: Use 1d EMA34 for trend direction and 12h Camarilla R3/S3 levels for entry.
Long when price crosses above EMA34 and touches 12h R3 level; short when price crosses below EMA34 and touches 12h S3 level.
Volume confirmation: current volume > 2x 20-period average volume to filter low-quality breakouts.
Designed for 12h timeframe to capture multi-day moves with tight entries (target: 12-37 trades/year).
Works in both bull and bear markets by following higher timeframe trend and using institutional pivot levels.
"""
name = "12h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_HT"
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
    
    # Get 1d data for EMA34 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = pd.Series(df_1d['close'])
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 12h data for Camarilla levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    
    # Calculate 12h Camarilla levels (R3, S3)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    pivot = (high_12h + low_12h + close_12h) / 3
    range_12h = high_12h - low_12h
    r3 = pivot + (range_12h * 1.1 / 4)
    s3 = pivot - (range_12h * 1.1 / 4)
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    
    # Volume filter: current volume > 2.0 * 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    start_idx = max(34, 20)  # Ensure sufficient warmup for EMA34 and volume
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 24 bars between trades (2 days on 12h TF) to reduce frequency
            if bars_since_exit < 24:
                continue
                
            # Long: price crosses above EMA34 and touches R3 level
            if (close[i] > ema_34_1d_aligned[i] and close[i-1] <= ema_34_1d_aligned[i-1] and 
                low[i] <= r3_aligned[i] and volume_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_exit = 0
            # Short: price crosses below EMA34 and touches S3 level
            elif (close[i] < ema_34_1d_aligned[i] and close[i-1] >= ema_34_1d_aligned[i-1] and 
                  high[i] >= s3_aligned[i] and volume_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit: price returns to opposite EMA34 side
            if position == 1 and close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            elif position == -1 and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals