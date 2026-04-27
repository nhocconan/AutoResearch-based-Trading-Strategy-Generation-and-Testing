#!/usr/bin/env python3
"""
12h Camarilla Pivot Reversal with 1d Trend and Volume Spike.
Long when price touches S3 and closes back above + daily trend up + volume spike.
Short when price touches R3 and closes back below + daily trend down + volume spike.
Exit when price reaches opposite Camarilla level (S1/R1) or trend reverses.
Designed for low frequency (10-25 trades/year) to minimize fee drag on 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period"""
    range_val = high - low
    if range_val <= 0:
        return close, close, close, close, close, close, close, close
    c = close
    h = high
    l = low
    r4 = c + ((h - l) * 1.500)
    r3 = c + ((h - l) * 1.250)
    r2 = c + ((h - l) * 1.166)
    r1 = c + ((h - l) * 1.083)
    s1 = c - ((h - l) * 1.083)
    s2 = c - ((h - l) * 1.166)
    s3 = c - ((h - l) * 1.250)
    s4 = c - ((h - l) * 1.500)
    return r4, r3, r2, r1, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter and Camarilla
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate EMA34 on daily close for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = np.full_like(close_1d, np.nan, dtype=np.float64)
    if len(close_1d) >= 34:
        ema_34_1d[33] = np.mean(close_1d[:34])
        for i in range(34, len(close_1d)):
            ema_34_1d[i] = (close_1d[i] * 2 + ema_34_1d[i-1] * 32) / 34
    
    # Calculate Camarilla levels for each daily bar
    r4_1d = np.full(len(df_1d), np.nan)
    r3_1d = np.full(len(df_1d), np.nan)
    r2_1d = np.full(len(df_1d), np.nan)
    r1_1d = np.full(len(df_1d), np.nan)
    s1_1d = np.full(len(df_1d), np.nan)
    s2_1d = np.full(len(df_1d), np.nan)
    s3_1d = np.full(len(df_1d), np.nan)
    s4_1d = np.full(len(df_1d), np.nan)
    
    for i in range(len(df_1d)):
        r4, r3, r2, r1, s1, s2, s3, s4 = calculate_camarilla(
            df_1d['high'].values[i],
            df_1d['low'].values[i],
            df_1d['close'].values[i]
        )
        r4_1d[i] = r4
        r3_1d[i] = r3
        r2_1d[i] = r2
        r1_1d[i] = r1
        s1_1d[i] = s1
        s2_1d[i] = s2
        s3_1d[i] = s3
        s4_1d[i] = s4
    
    # Align daily indicators to 12h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Volume filter: volume > 1.8x average (adapted for 12h)
    vol_ma_20 = np.full_like(volume, np.nan, dtype=np.float64)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need EMA (34) + volume MA (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        
        # Current indicators
        ema_34 = ema_34_aligned[i]
        r3 = r3_aligned[i]
        r1 = r1_aligned[i]
        s3 = s3_aligned[i]
        s1 = s1_aligned[i]
        
        # Volume filter: volume > 1.8x average
        vol_filter = vol_now > 1.8 * vol_ma_20[i]
        
        if position == 0:
            # Long: price touches S3 and closes back above + daily trend up + volume spike
            if (price_now <= s3 * 1.002 and close[i] > s3 and  # touches S3 and closes above
                close[i] > ema_34 and  # daily trend up
                vol_filter):
                signals[i] = size
                position = 1
            # Short: price touches R3 and closes back below + daily trend down + volume spike
            elif (price_now >= r3 * 0.998 and close[i] < r3 and  # touches R3 and closes below
                  close[i] < ema_34 and  # daily trend down
                  vol_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches S1 (take profit) or trend turns down
            if price_now >= s1 or close[i] < ema_34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price reaches R1 (take profit) or trend turns up
            if price_now <= r1 or close[i] > ema_34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R3S3_Reversal_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0