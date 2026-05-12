#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1. Daily Camarilla pivot levels (from previous day)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: pivot = (H+L+C)/3, width = H-L
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    width_1d = high_1d - low_1d
    r1_1d = close_1d + (width_1d * 1.1 / 12)  # R1 = C + (H-L)*1.1/12
    s1_1d = close_1d - (width_1d * 1.1 / 12)  # S1 = C - (H-L)*1.1/12
    
    # Align Camarilla levels to 4h
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # 2. Daily trend: close above/below EMA34
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    daily_uptrend = close_1d > ema34_1d
    daily_downtrend = close_1d < ema34_1d
    daily_uptrend_aligned = align_htf_to_ltf(prices, df_1d, daily_uptrend)
    daily_downtrend_aligned = align_htf_to_ltf(prices, df_1d, daily_downtrend)
    
    # 3. Volume spike: volume > 1.5 * 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # ensure volume MA has enough data
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: daily uptrend + price breaks above R1 + volume spike
            if (daily_uptrend_aligned[i] and 
                close[i] > r1_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: daily downtrend + price breaks below S1 + volume spike
            elif (daily_downtrend_aligned[i] and 
                  close[i] < s1_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below S1 or daily trend changes
            if close[i] < s1_1d_aligned[i] or not daily_uptrend_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above R1 or daily trend changes
            if close[i] > r1_1d_aligned[i] or not daily_downtrend_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals