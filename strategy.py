# 12h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike
# Hypothesis: 12h timeframe reduces trade frequency to avoid fee drag. Uses weekly trend filter (1w EMA34) for stronger trend bias,
# daily Camarilla R3/S3 levels for entry/exit, and volume spike confirmation. Weekly trend filter should work in both bull and bear
# by only taking trades in the direction of the higher timeframe trend, reducing whipsaw.
# Target: 50-150 total trades over 4 years (12-37/year) to stay within fee drag limits.

#!/usr/bin/env python3
name = "12h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots (same as before)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # EMA(34) on 1w for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate Camarilla pivot levels for previous day (R3, S3)
    p = (high_1d + low_1d + close_1d) / 3
    r3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    s3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    
    # Align Camarilla levels to 12h
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume spike on 12h: current volume > 2.0x 30-period average (slightly higher threshold for fewer trades)
    vol_avg = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R3 + 1w trend up + volume spike
            if (close[i] > r3_aligned[i] and 
                close[i] > ema34_1w_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 + 1w trend down + volume spike
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema34_1w_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below S3
            if close[i] < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above R3
            if close[i] > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals