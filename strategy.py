#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data (HTF) once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily ATR (14-period)
    tr = np.zeros(len(df_1d))
    tr[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(df_1d)):
        tr[i] = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - close_1d[i-1]),
            abs(low_1d[i] - close_1d[i-1])
        )
    
    atr_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 14:
        atr_1d[13] = np.mean(tr[:14])
        for i in range(14, len(df_1d)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Calculate daily ATR ratio (ATR / Close) for volatility filter
    atr_ratio_1d = np.full(len(df_1d), np.nan)
    for i in range(len(df_1d)):
        if not np.isnan(atr_1d[i]) and close_1d[i] > 0:
            atr_ratio_1d[i] = atr_1d[i] / close_1d[i]
    
    atr_ratio_6h = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    # Calculate 6-hour Donchian channels (20-period)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            donch_high[i] = np.max(high[i-19:i+1])
            donch_low[i] = np.min(low[i-19:i+1])
    
    # Calculate daily volatility filter (ATR ratio > 0.015 = 1.5%)
    vol_filter_6h = np.zeros(n)
    for i in range(len(atr_ratio_6h)):
        if not np.isnan(atr_ratio_6h[i]) and atr_ratio_6h[i] > 0.015:
            vol_filter_6h[i] = 1.0
    
    # Calculate daily pivot levels from previous day
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    prev_range = prev_high - prev_low
    
    # Camarilla-style pivot levels (R3/S3)
    r3 = prev_close + (prev_range * 1.1 / 4)
    s3 = prev_close - (prev_range * 1.1 / 4)
    
    # Align to 6h timeframe
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(60, n):
        # Skip if any critical data is NaN
        if (np.isnan(atr_ratio_6h[i]) or
            np.isnan(donch_high[i]) or
            np.isnan(donch_low[i]) or
            np.isnan(r3_6h[i]) or
            np.isnan(s3_6h[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR ratio < 1.5%)
        if vol_filter_6h[i] < 0.5:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above 6h Donchian high AND above S3
            if close[i] > donch_high[i] and close[i] > s3_6h[i]:
                position = 1
                signals[i] = position_size
            # Short: Price breaks below 6h Donchian low AND below R3
            elif close[i] < donch_low[i] and close[i] < r3_6h[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price falls back below 6h Donchian low OR below S3
            if close[i] < donch_low[i] or close[i] < s3_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price rises back above 6h Donchian high OR above R3
            if close[i] > donch_high[i] or close[i] > r3_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_Camarilla_R3S3_Donchian_VolFilter"
timeframe = "6h"
leverage = 1.0