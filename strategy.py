#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_Volume_Enhanced
Hypothesis: Use 12h chart with Camarilla R3/S3 levels from daily pivot for breakout entries,
filtered by 1-day EMA trend and volume spikes. Exit on trend reversal or opposite level break.
Designed for low trade frequency (15-30/year) with strong trend filtering to work in both bull and bear markets.
Timeframe: 12h, uses 1d HTF for Camarilla levels and trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume_Enhanced"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1-day data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # Typical price = (H + L + C) / 3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    range_val = df_1d['high'] - df_1d['low']
    
    # Camarilla levels: R3 = PP + 1.1 * (H-L)/2, S3 = PP - 1.1 * (H-L)/2
    # Actually standard Camarilla: R3 = PP + 1.1*(H-L)/2, S3 = PP - 1.1*(H-L)/2
    # But we'll use the common definition: R3 = C + 1.1*(H-L)/2, S3 = C - 1.1*(H-L)/2
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    camarilla_pp = (high_1d + low_1d + close_1d) / 3
    camarilla_range = high_1d - low_1d
    camarilla_r3 = close_1d + 1.1 * camarilla_range / 2
    camarilla_s3 = close_1d - 1.1 * camarilla_range / 2
    
    # Align Camarilla levels to 12h chart (previous day's levels)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # 1-day EMA trend filter (21-period)
    ema_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume spike: current volume > 2.0 * 24-period average (2 days of 12h data)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(24, n):  # Wait for warmup
        # Skip if any critical value is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 + uptrend + volume spike
            if (close[i] > r3_aligned[i] and 
                ema_1d_aligned[i] > ema_1d_aligned[i-1] and 
                volume[i] > 2.0 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 + downtrend + volume spike
            elif (close[i] < s3_aligned[i] and 
                  ema_1d_aligned[i] < ema_1d_aligned[i-1] and 
                  volume[i] > 2.0 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below S3 or trend turns down
            if close[i] < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif ema_1d_aligned[i] < ema_1d_aligned[i-1]:  # Trend turns down
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above R3 or trend turns up
            if close[i] > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif ema_1d_aligned[i] > ema_1d_aligned[i-1]:  # Trend turns up
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals