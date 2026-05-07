#!/usr/bin/env python3
"""
6H_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_v3
Hypothesis: On 6b timeframe, use daily Camarilla R3/S3 levels for breakout entries.
Long when price closes above R3 with volume spike and daily close above EMA34 (bullish trend).
Short when price closes below S3 with volume spike and daily close below EMA34 (bearish trend).
Volume confirmation: current volume > 2x 20-period average volume.
Exit when price returns to the opposite Camarilla level (R3 for longs, S3 for shorts).
Designed to capture strong momentum moves in both bull and bear markets with filtered entries.
"""
name = "6H_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_v3"
timeframe = "6h"
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
    
    # Get 1d data for Camarilla levels and EMA34 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (R3, S3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r3 = pivot + (range_1d * 1.1 / 4)  # R3 level
    s3 = pivot - (range_1d * 1.1 / 4)  # S3 level
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate 1d EMA34 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema34 = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34)
    
    # Volume filter: current volume > 2x 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    start_idx = max(35, 20)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 10 bars between trades (~2.5 days on 6h TF) to reduce frequency
            if bars_since_exit < 10:
                continue
                
            # Long: price closes above R3 with volume spike and daily close above EMA34
            if (close[i] > r3_aligned[i] and volume_filter[i] and 
                close_1d.iloc[-1] > ema34[-1] if len(close_1d) > 0 else False):
                # Actually need to check current daily close vs its EMA
                # Find the current 1d bar index for this 6h bar
                # Since we're using aligned arrays, we can check the trend condition directly
                if close[i] > r3_aligned[i] and volume_filter[i] and close[i] > ema34_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    bars_since_exit = 0
            # Short: price closes below S3 with volume spike and daily close below EMA34
            elif (close[i] < s3_aligned[i] and volume_filter[i] and close[i] < ema34_aligned[i]):
                signals[i] = -0.25
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit: price returns to opposite Camarilla level
            if position == 1 and close[i] < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            elif position == -1 and close[i] > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals