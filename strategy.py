#!/usr/bin/env python3
"""
4H_Camarilla_R3_S3_Breakout_1dEMA34_Trend_Filter_v2
Hypothesis: Trade breakouts of 1d Camarilla R3/S3 levels with 1d EMA34 trend filter and volume confirmation.
Long when price breaks above R3 and 1d EMA34 is rising; short when price breaks below S3 and 1d EMA34 is falling.
Volume filter: current volume > 1.5x 20-period average volume.
This strategy targets high-probability breakouts in trending markets while avoiding false signals in chop.
Designed for 4h timeframe with 1d HTF for trend and pivot levels.
"""
name = "4H_Camarilla_R3_S3_Breakout_1dEMA34_Trend_Filter_v2"
timeframe = "4h"
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
    
    # Get 1d data for Camarilla levels and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
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
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume filter: current volume > 1.5 * 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    start_idx = max(34, 20)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 8 bars between trades (1.33 days on 4h TF) to reduce frequency
            if bars_since_exit < 8:
                continue
                
            # Long: price breaks above R3 with rising EMA34 and volume
            if (close[i] > r3_aligned[i] and close[i-1] <= r3_aligned[i-1] and 
                ema_34_aligned[i] > ema_34_aligned[i-1] and volume_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_exit = 0
            # Short: price breaks below S3 with falling EMA34 and volume
            elif (close[i] < s3_aligned[i] and close[i-1] >= s3_aligned[i-1] and 
                  ema_34_aligned[i] < ema_34_aligned[i-1] and volume_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit: price returns to opposite S3/R3 level
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