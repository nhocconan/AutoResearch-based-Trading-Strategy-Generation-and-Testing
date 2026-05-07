#!/usr/bin/env python3
name = "1d_Camarilla_R3S3_Breakout_1wTrend_Volume"
timeframe = "1d"
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
    
    # Get 1w data for weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # 1d data for Camarilla levels (same timeframe)
    df_1d = prices  # since timeframe is 1d, prices is already 1d
    
    # 1d Camarilla R3/S3 levels (stronger breakout levels)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    range_1d = high_1d - low_1d
    r3_1d = close_1d + 1.1666 * range_1d * 1.1 / 2
    s3_1d = close_1d - 1.1666 * range_1d * 1.1 / 2
    
    # 1w EMA20 for weekly trend filter (weekly close EMA)
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 1w EMA to 1d timeframe
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # for EMA20 and volume average
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(r3_1d[i]) or np.isnan(s3_1d[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 + weekly uptrend + volume
            if close[i] > r3_1d[i] and close[i] > ema_20_1w_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 + weekly downtrend + volume
            elif close[i] < s3_1d[i] and close[i] < ema_20_1w_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price crosses back through the opposite S3/R3 level
            if position == 1:
                if close[i] < s3_1d[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > r3_1d[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals