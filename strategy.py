#!/usr/bin/env python3
name = "1h_Camarilla_R3S3_Breakout_4hTrend"
timeframe = "1h"
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
    
    # Calculate 4h Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 1:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla: R3, S3, R4, S4
    pivot_4h = (high_4h + low_4h + close_4h) / 3
    range_4h = high_4h - low_4h
    
    R3_4h = close_4h + range_4h * 1.1 / 4
    S3_4h = close_4h - range_4h * 1.1 / 4
    R4_4h = close_4h + range_4h * 1.1 / 2
    S4_4h = close_4h - range_4h * 1.1 / 2
    
    # Align to 1h
    R3_4h_aligned = align_htf_to_ltf(prices, df_4h, R3_4h)
    S3_4h_aligned = align_htf_to_ltf(prices, df_4h, S3_4h)
    R4_4h_aligned = align_htf_to_ltf(prices, df_4h, R4_4h)
    S4_4h_aligned = align_htf_to_ltf(prices, df_4h, S4_4h)
    
    # 4h EMA50 trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1h volume filter (20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(R3_4h_aligned[i]) or np.isnan(S3_4h_aligned[i]) or
            np.isnan(R4_4h_aligned[i]) or np.isnan(S4_4h_aligned[i]) or
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above R3 with volume and 4h uptrend
            if (close[i] > R3_4h_aligned[i] and 
                volume[i] > 1.5 * vol_ma20[i] and 
                close[i] > ema_50_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below S3 with volume and 4h downtrend
            elif (close[i] < S3_4h_aligned[i] and 
                  volume[i] > 1.5 * vol_ma20[i] and 
                  close[i] < ema_50_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: Price breaks below S3 or 4h trend reverses
            if close[i] < S3_4h_aligned[i] or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: Price breaks above R3 or 4h trend reverses
            if close[i] > R3_4h_aligned[i] or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals