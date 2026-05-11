#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_Breakout_VolumeTrend"
timeframe = "4h"
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
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    # Using previous day's OHLC (shifted by 1 to avoid look-ahead)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate pivot and support/resistance levels
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # Camarilla levels
    R1 = pivot + (range_val * 1.1 / 12)
    R2 = pivot + (range_val * 1.1 / 6)
    R3 = pivot + (range_val * 1.1 / 4)
    R4 = pivot + (range_val * 1.1 / 2)
    S1 = pivot - (range_val * 1.1 / 12)
    S2 = pivot - (range_val * 1.1 / 6)
    S3 = pivot - (range_val * 1.1 / 4)
    S4 = pivot - (range_val * 1.1 / 2)
    
    # Align levels to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    
    # Weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_trend = weekly_close > weekly_ema
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend)
    
    # Volume confirmation - 20-period average
    vol_ma = np.zeros(n)
    for i in range(n):
        if i < 20:
            vol_ma[i] = np.mean(volume[:i+1]) if i > 0 else 0
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(weekly_trend_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with volume and weekly uptrend
            if (close[i] > R1_aligned[i] and 
                volume[i] > 1.5 * vol_ma[i] and 
                weekly_trend_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume and weekly downtrend
            elif (close[i] < S1_aligned[i] and 
                  volume[i] > 1.5 * vol_ma[i] and 
                  not weekly_trend_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S1 or weekly trend changes
            if (close[i] < S1_aligned[i] or not weekly_trend_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R1 or weekly trend changes
            if (close[i] > R1_aligned[i] or weekly_trend_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals