#!/usr/bin/env python3
name = "4H_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
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
    
    # Get daily data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for the previous day
    # H = previous day high, L = previous day low, C = previous day close
    H = high_1d[:-1]  # yesterday's high
    L = low_1d[:-1]   # yesterday's low
    C = close_1d[:-1] # yesterday's close
    
    # Camarilla R3 and S3 levels
    R3 = C + (H - L) * 1.1 / 2
    S3 = C - (H - L) * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # 1-day EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or np.isnan(ema34_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 2.0x 24-period average volume (~1 day)
        if i >= 24:
            avg_volume = np.mean(volume[i-24:i])
            volume_confirm = volume[i] > avg_volume * 2.0
        else:
            volume_confirm = False
        
        if position == 0:
            # Enter long: price breaks above R3 + uptrend + volume confirmation
            if close[i] > R3_aligned[i] and close[i] > ema34_1d_aligned[i] and volume_confirm:
                signals[i] = 0.30
                position = 1
            # Enter short: price breaks below S3 + downtrend + volume confirmation
            elif close[i] < S3_aligned[i] and close[i] < ema34_1d_aligned[i] and volume_confirm:
                signals[i] = -0.30
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S3 or trend reverses
            if close[i] < S3_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Exit short: price breaks above R3 or trend reverses
            if close[i] > R3_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals