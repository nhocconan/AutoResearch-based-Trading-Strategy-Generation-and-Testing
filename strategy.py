#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_12hEMA20_Trend_Volume_v1
Hypothesis: Use 12h EMA20 as a stronger trend filter than daily EMA50, combined with tight 1d Camarilla R1/S1 breakouts and volume > 1.3x average. Designed for fewer, higher-quality trades (target 15-25/year) to reduce fee drag. Works in bull markets (breakouts with trend) and bear markets (fades from extremes with trend confirmation).
"""
name = "4h_Camarilla_R1S1_Breakout_12hEMA20_Trend_Volume_v1"
timeframe = "4h"
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
    
    # Get 12h and 1d data
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # 1-day OHLC for Camarilla pivot (R1/S1 = tight breakout levels)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r1_1d = close_1d + (range_1d * 1.1 / 6)   # R1
    s1_1d = close_1d - (range_1d * 1.1 / 6)   # S1
    
    # Align Camarilla levels to 4h
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # 12-hour EMA20 for trend filter (more responsive than daily EMA50)
    ema_20_12h = pd.Series(df_12h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # Volume filter: current volume > 1.3 * 50-period average
    vol_avg = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_filter = volume > (vol_avg * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for EMA20 and volume average
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(ema_20_12h_aligned[i]) or np.isnan(vol_avg[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1 + 12h uptrend + volume filter
            if (close[i] > r1_1d_aligned[i] and 
                close[i] > ema_20_12h_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + 12h downtrend + volume filter
            elif (close[i] < s1_1d_aligned[i] and 
                  close[i] < ema_20_12h_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price returns to opposite Camarilla level (S1 for long, R1 for short)
            if position == 1:
                if close[i] <= s1_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] >= r1_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals