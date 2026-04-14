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
    
    # Load daily data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily ATR (14-period) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    high_low = high_1d - low_1d
    high_close = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    low_close = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    
    atr_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 14:
        atr_1d[13] = np.mean(tr[:14])
        for i in range(14, len(df_1d)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Align daily ATR to 12h timeframe
    atr_12h_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate daily pivot points (classic)
    pivot = (high_1d + low_1d + close_1d) / 3
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    r2 = pivot + (high_1d - low_1d)
    s2 = pivot - (high_1d - low_1d)
    r3 = high_1d + 2 * (pivot - low_1d)
    s3 = low_1d - 2 * (high_1d - pivot)
    
    # Align pivot levels to 12h timeframe (constant values for the day)
    pivot_12h = align_htf_to_ltf(prices, df_1d, pivot)
    r1_12h = align_htf_to_ltf(prices, df_1d, r1)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1)
    r2_12h = align_htf_to_ltf(prices, df_1d, r2)
    s2_12h = align_htf_to_ltf(prices, df_1d, s2)
    r3_12h = align_htf_to_ltf(prices, df_1d, r3)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(14, n):
        # Skip if any critical data is NaN
        if (np.isnan(atr_12h_aligned[i]) or
            np.isnan(pivot_12h[i]) or
            np.isnan(r1_12h[i]) or
            np.isnan(s1_12h[i]) or
            np.isnan(r2_12h[i]) or
            np.isnan(s2_12h[i]) or
            np.isnan(r3_12h[i]) or
            np.isnan(s3_12h[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 0.3% of price)
        if atr_12h_aligned[i] / close[i] < 0.003:
            signals[i] = 0.0
            continue
        
        # Skip low volume periods (volume < 50% of previous 20 periods average)
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
            if volume[i] < 0.5 * vol_ma:
                signals[i] = 0.0
                continue
        
        if position == 0:
            # Long: Price breaks below S3 (extreme rejection) AND closes back above S3
            if low[i] <= s3_12h[i] and close[i] > s3_12h[i]:
                position = 1
                signals[i] = position_size
            # Short: Price breaks above R3 (extreme rejection) AND closes back below R3
            elif high[i] >= r3_12h[i] and close[i] < r3_12h[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price breaks below S3 again or reaches R2 (mean reversion target)
            if low[i] <= s3_12h[i] or close[i] >= r2_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price breaks above R3 again or reaches S2 (mean reversion target)
            if high[i] >= r3_12h[i] or close[i] <= s2_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Pivot_S3R3_Breakout_Volume_Filter_v2"
timeframe = "12h"
leverage = 1.0