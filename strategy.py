#!/usr/bin/env python3
name = "6h_Alligator_R3S3_Breakout_1dTrend"
timeframe = "6h"
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
    
    # Williams Alligator (13,8,5) smoothed with SMMA
    def smma(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        sma = np.mean(arr[:period])
        result[period-1] = sma
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Alligator alignment: bullish when lips > teeth > jaw
    bullish_align = (lips > teeth) & (teeth > jaw)
    bearish_align = (lips < teeth) & (teeth < jaw)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 50-period EMA on 1d
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get weekly data for pivot levels (R3/S3)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's OHLC)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Pivot point = (H + L + C) / 3
    pp = (high_1w + low_1w + close_1w) / 3.0
    # R3 = H + 2*(PP - L)
    r3 = high_1w + 2 * (pp - low_1w)
    # S3 = L - 2*(H - PP)
    s3 = low_1w - 2 * (high_1w - pp)
    
    # Align weekly levels to 6h
    r3_6h = align_htf_to_ltf(prices, df_1w, r3)
    s3_6h = align_htf_to_ltf(prices, df_1w, s3)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for Alligator and MA
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Bullish Alligator alignment + price > R3 + price > 1d EMA50 + volume
            if (bullish_align[i] and close[i] > r3_6h[i] and 
                close[i] > ema_50_1d_aligned[i] and volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator alignment + price < S3 + price < 1d EMA50 + volume
            elif (bearish_align[i] and close[i] < s3_6h[i] and 
                  close[i] < ema_50_1d_aligned[i] and volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bearish Alligator alignment OR price < S3
            if bearish_align[i] or close[i] < s3_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bullish Alligator alignment OR price > R3
            if bullish_align[i] or close[i] > r3_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals