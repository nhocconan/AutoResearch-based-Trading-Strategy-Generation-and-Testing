#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_Camarilla_Pivot_Volume_Trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 12h data for Camarilla pivots
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 12h Camarilla pivot levels
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Pivot = (H + L + C) / 3
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    # Range = H - L
    range_12h = high_12h - low_12h
    # Camarilla levels
    r4_12h = close_12h + range_12h * 1.1 / 2
    r3_12h = close_12h + range_12h * 1.1 / 4
    r2_12h = close_12h + range_12h * 1.1 / 6
    r1_12h = close_12h + range_12h * 1.1 / 12
    s1_12h = close_12h - range_12h * 1.1 / 12
    s2_12h = close_12h - range_12h * 1.1 / 6
    s3_12h = close_12h - range_12h * 1.1 / 4
    s4_12h = close_12h - range_12h * 1.1 / 2
    
    # Align 12h Camarilla levels to 4h timeframe
    pivot_12h_aligned = align_htf_to_ltf(prices, df_12h, pivot_12h)
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    r4_12h_aligned = align_htf_to_ltf(prices, df_12h, r4_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    s4_12h_aligned = align_htf_to_ltf(prices, df_12h, s4_12h)
    
    # Volume confirmation - 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1.0)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(pivot_12h_aligned[i]) or 
            np.isnan(r3_12h_aligned[i]) or np.isnan(r4_12h_aligned[i]) or
            np.isnan(s3_12h_aligned[i]) or np.isnan(s4_12h_aligned[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above R3 + above daily EMA50 + volume confirmation
            if (close[i] > r3_12h_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and
                vol_ratio[i] > 1.5):
                # Avoid extreme extension beyond R4
                if close[i] <= r4_12h_aligned[i] * 1.02:
                    signals[i] = 0.25
                    position = 1
            # Short: price below S3 + below daily EMA50 + volume confirmation
            elif (close[i] < s3_12h_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and
                  vol_ratio[i] > 1.5):
                # Avoid extreme extension beyond S4
                if close[i] >= s4_12h_aligned[i] * 0.98:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price below R3 OR below daily EMA50
            if close[i] < r3_12h_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price above S3 OR above daily EMA50
            if close[i] > s3_12h_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals