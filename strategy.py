#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_Volume
Breakout at Camarilla R1/S1 levels with 1d EMA50 trend filter and volume confirmation.
Long when price breaks above R1 with volume > 1.5x average and above 1d EMA50.
Short when price breaks below S1 with volume > 1.5x average and below 1d EMA50.
Exit when price returns to pivot (HLC/3) or trend filter fails.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate average volume for confirmation (20-period)
    vol_avg = np.full(n, np.nan)
    for i in range(19, n):
        vol_avg[i] = np.mean(volume[i-19:i+1])
    
    # Get 1d data for pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_pivot = np.full(len(close_1d), np.nan)
    camarilla_r1 = np.full(len(close_1d), np.nan)
    camarilla_s1 = np.full(len(close_1d), np.nan)
    
    for i in range(len(close_1d)):
        camarilla_pivot[i] = (high_1d[i] + low_1d[i] + close_1d[i]) / 3
        camarilla_r1[i] = camarilla_pivot[i] + 1.1 * (high_1d[i] - low_1d[i]) / 12
        camarilla_s1[i] = camarilla_pivot[i] - 1.1 * (high_1d[i] - low_1d[i]) / 12
    
    # Align Camarilla levels to 4h timeframe
    pivot_4h = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    r1_4h = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Get 1d EMA50 for trend filter
    ema_1d_period = 50
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= ema_1d_period:
        ema_1d[ema_1d_period - 1] = np.mean(close_1d[:ema_1d_period])
        for i in range(ema_1d_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * (2 / (ema_1d_period + 1)) + 
                         ema_1d[i - 1] * (1 - (2 / (ema_1d_period + 1))))
    
    # Align 1d EMA50 to 4h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need volume average and EMA
    start_idx = max(19, ema_1d_period - 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(vol_avg[i]) or np.isnan(pivot_4h[i]) or 
            np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or 
            np.isnan(ema_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        pivot = pivot_4h[i]
        r1 = r1_4h[i]
        s1 = s1_4h[i]
        ema1d = ema_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume confirmation and above 1d EMA50
            if (price > r1 and vol > 1.5 * vol_avg[i] and price > ema1d):
                signals[i] = size
                position = 1
            # Short: price breaks below S1 with volume confirmation and below 1d EMA50
            elif (price < s1 and vol > 1.5 * vol_avg[i] and price < ema1d):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to pivot or trend fails
            if (price <= pivot) or (price < ema1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to pivot or trend fails
            if (price >= pivot) or (price > ema1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0