#!/usr/bin/env python3
# 12h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike
# Hypothesis: Buy/sell at Camarilla R3/S3 levels on 12h timeframe with daily trend and volume confirmation.
# Works in bull/bear by following daily trend and using volume to confirm institutional interest.
# Target: 20-40 trades/year per symbol.

name = "12h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
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
    
    # Daily Camarilla levels (R3, S3)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot and Camarilla levels
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r3 = pivot + (range_1d * 1.1 / 2)
    s3 = pivot - (range_1d * 1.1 / 2)
    
    # Align Camarilla levels to 12h
    r3_12h = align_htf_to_ltf(prices, df_1d, r3)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3)
    
    # Daily trend filter (EMA50)
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    daily_uptrend = close_1d > ema50_1d
    daily_downtrend = close_1d < ema50_1d
    
    # Align daily trend to 12h
    daily_uptrend_12h = align_htf_to_ltf(prices, df_1d, daily_uptrend.astype(float))
    daily_downtrend_12h = align_htf_to_ltf(prices, df_1d, daily_downtrend.astype(float))
    
    # Volume spike detection (20-period average)
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r3_12h[i]) or np.isnan(s3_12h[i]) or 
            np.isnan(daily_uptrend_12h[i]) or np.isnan(daily_downtrend_12h[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_spike = vol_ratio > 2.0
        
        daily_up = daily_uptrend_12h[i] > 0.5
        daily_down = daily_downtrend_12h[i] > 0.5
        
        if position == 0:
            # Enter long: daily uptrend + price breaks above R3 + volume spike
            if daily_up and close[i] > r3_12h[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: daily downtrend + price breaks below S3 + volume spike
            elif daily_down and close[i] < s3_12h[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price returns below R3 or trend changes
            if close[i] < r3_12h[i] or not daily_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price returns above S3 or trend changes
            if close[i] > s3_12h[i] or not daily_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals