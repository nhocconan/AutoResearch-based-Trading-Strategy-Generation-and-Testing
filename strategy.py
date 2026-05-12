#!/usr/bin/env python3
name = "6h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "6h"
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
    
    # Load daily data once for trend filter and Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    # Daily EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Daily OHLC for Camarilla R3 and S3 (previous day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    # Calculate Camarilla R3 and S3 for previous day
    p = (high_1d + low_1d + close_1d_vals) / 3
    r3 = close_1d_vals + (high_1d - low_1d) * 1.1 / 4 * 3  # R3: C + (H-L)*1.1*3/4
    s3 = close_1d_vals - (high_1d - low_1d) * 1.1 / 4 * 3  # S3: C - (H-L)*1.1*3/4
    
    # Align Camarilla levels to 6h (wait for daily close)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_avg)
    
    # EMA slope: positive when current EMA > previous EMA
    ema_slope = ema34_1d_aligned > np.roll(ema34_1d_aligned, 1)
    ema_slope[0] = False  # first value invalid
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R3 + daily trend up + volume spike + EMA slope up
            if (close[i] > r3_aligned[i] and 
                close[i] > ema34_1d_aligned[i] and 
                vol_spike[i] and 
                ema_slope[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 + daily trend down + volume spike + EMA slope down
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema34_1d_aligned[i] and 
                  vol_spike[i] and 
                  not ema_slope[i]):
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