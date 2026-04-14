#!/usr/bin/env python3
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
    
    # Load daily data (HTF) once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR (14-period) for volatility filter
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
    
    atr_12h = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 12-hour Donchian channels (20-period)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            donch_high[i] = np.max(high[i-19:i+1])
            donch_low[i] = np.min(low[i-19:i+1])
    
    # Calculate daily pivot levels (previous day)
    prev_high = np.concatenate([[high_1d[0]], high_1d[:-1]])
    prev_low = np.concatenate([[low_1d[0]], low_1d[:-1]])
    prev_close = np.concatenate([[close_1d[0]], close_1d[:-1]])
    prev_range = prev_high - prev_low
    
    # Camarilla-style pivot levels (R3/S3)
    r3_1d = prev_close + (prev_range * 1.1 / 4)
    s3_1d = prev_close - (prev_range * 1.1 / 4)
    
    # Align to 12h timeframe
    r3_12h = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if (np.isnan(atr_12h[i]) or
            np.isnan(donch_high[i]) or
            np.isnan(donch_low[i]) or
            np.isnan(r3_12h[i]) or
            np.isnan(s3_12h[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 0.5% of price)
        if atr_12h[i] < 0.005 * close[i]:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above 12h Donchian high AND above S3
            if close[i] > donch_high[i] and close[i] > s3_12h[i]:
                position = 1
                signals[i] = position_size
            # Short: Price breaks below 12h Donchian low AND below R3
            elif close[i] < donch_low[i] and close[i] < r3_12h[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price falls back below 12h Donchian low OR below S3
            if close[i] < donch_low[i] or close[i] < s3_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price rises back above 12h Donchian high OR above R3
            if close[i] > donch_high[i] or close[i] > r3_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Camarilla_R3S3_Breakout_Donchian_v2"
timeframe = "12h"
leverage = 1.0