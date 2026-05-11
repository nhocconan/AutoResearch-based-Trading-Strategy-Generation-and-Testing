#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_Breakout_Trend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla and trend
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Previous day's high, low, close for Camarilla levels
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    prev_close = df_1d['close'].values
    
    # Calculate Camarilla levels (R1, S1)
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    camarilla_r1 = prev_close + 1.1 * (prev_high - prev_low) / 12
    camarilla_s1 = prev_close - 1.1 * (prev_high - prev_low) / 12
    
    # Daily EMA34 for trend filter
    daily_close = df_1d['close'].values
    ema34_d = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    daily_trend = daily_close > ema34_d
    
    # Volume spike detection (2x 20-period average)
    vol_ma20 = np.zeros(n)
    for i in range(n):
        if i < 20:
            vol_ma20[i] = np.mean(volume[:i+1]) if i > 0 else 0
        else:
            vol_ma20[i] = np.mean(volume[i-19:i+1])
    
    # Align Camarilla levels and trend to 4h
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    daily_trend_aligned = align_htf_to_ltf(prices, df_1d, daily_trend)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 34)
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(daily_trend_aligned[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 + uptrend + volume spike
            if (close[i] > camarilla_r1_aligned[i] and 
                daily_trend_aligned[i] and 
                volume[i] > 2.0 * vol_ma20[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + downtrend + volume spike
            elif (close[i] < camarilla_s1_aligned[i] and 
                  not daily_trend_aligned[i] and 
                  volume[i] > 2.0 * vol_ma20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S1 or trend changes
            if (close[i] < camarilla_s1_aligned[i] or not daily_trend_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R1 or trend changes
            if (close[i] > camarilla_r1_aligned[i] or daily_trend_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals