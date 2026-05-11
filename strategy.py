#!/usr/bin/env python3
name = "12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot, trend, and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Camarilla pivot levels from 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    range_1d = high_1d - low_1d
    r3 = close_1d + range_1d * 1.1 / 4
    s3 = close_1d - range_1d * 1.1 / 4
    
    # Align Camarilla levels to 12h
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # 1d trend: EMA 34
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike detection: volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(ema34_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close > R3 AND price > EMA34 AND volume spike
            if (close[i] > r3_aligned[i] and 
                close[i] > ema34_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Close < S3 AND price < EMA34 AND volume spike
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema34_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close < S3 OR price < EMA34
            if (close[i] < s3_aligned[i] or 
                close[i] < ema34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: Close > R3 OR price > EMA34
            if (close[i] > r3_aligned[i] or 
                close[i] > ema34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals