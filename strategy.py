#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla R3/S3 Fade with 12h Volume Spike and 1d ADX Regime Filter
- Uses Camarilla pivot levels from daily data: fade at R3/S3 levels
- Only trade when 12h volume > 1.5x 20-period average (confirms institutional interest)
- 1d ADX > 25 ensures we're in a trending environment where fading extremes works
- Designed for 6h timeframe targeting 15-25 trades/year (60-100 over 4 years)
- Works in both bull and bear markets by fading overextended moves in trending regimes
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla_pivot(high, low, close):
    """Calculate Camarilla pivot levels"""
    pivot = (high + low + close) / 3
    range_val = high - low
    r3 = close + range_val * 1.1 / 2
    s3 = close - range_val * 1.1 / 2
    r4 = close + range_val * 1.1 / 2 * 2
    s4 = close - range_val * 1.1 / 2 * 2
    return r3, s3, r4, s4

def calculate_adx(high, low, close, period=14):
    """Calculate ADX indicator"""
    plus_dm = np.zeros_like(high)
    minus_dm = np.zeros_like(high)
    tr = np.zeros_like(high)
    
    for i in range(1, len(high)):
        plus_dm[i] = max(0, high[i] - high[i-1])
        minus_dm[i] = max(0, low[i-1] - low[i])
        if plus_dm[i] < minus_dm[i]:
            plus_dm[i] = 0
        if minus_dm[i] < plus_dm[i]:
            minus_dm[i] = 0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Wilder's smoothing
    atr = np.zeros_like(tr)
    atr[period] = np.mean(tr[1:period+1])
    for i in range(period+1, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    plus_di = np.zeros_like(high)
    minus_di = np.zeros_like(high)
    dx = np.zeros_like(high)
    
    for i in range(period, len(high)):
        if atr[i] > 0:
            plus_di[i] = 100 * (np.mean(plus_dm[i-period+1:i+1]) / atr[i])
            minus_di[i] = 100 * (np.mean(minus_dm[i-period+1:i+1]) / atr[i])
            if plus_di[i] + minus_di[i] > 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    adx = np.zeros_like(dx)
    adx[2*period-1] = np.mean(dx[period:2*period])
    for i in range(2*period, len(dx)):
        adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot (R3/S3) and ADX
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    r3_1d = np.zeros_like(close_1d)
    s3_1d = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        r3, s3, _, _ = calculate_camarilla_pivot(high_1d[i], low_1d[i], close_1d[i])
        r3_1d[i] = r3
        s3_1d[i] = s3
    
    # Calculate 1d ADX
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, period=14)
    
    # Align HTF indicators to 6h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation: > 1.5x 20-period average on 12h
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    volume_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # need 1d ADX, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check volume confirmation
        volume_ok = volume[i] > 1.5 * vol_ma_12h_aligned[i]
        
        # Check ADX regime filter (trending market)
        adx_ok = adx_1d_aligned[i] > 25
        
        if position == 0:
            # Long fade: price touches or goes below S3 and reverses up
            if (low[i] <= s3_1d_aligned[i] and close[i] > s3_1d_aligned[i] and 
                volume_ok and adx_ok):
                signals[i] = 0.25
                position = 1
            # Short fade: price touches or goes above R3 and reverses down
            elif (high[i] >= r3_1d_aligned[i] and close[i] < r3_1d_aligned[i] and 
                  volume_ok and adx_ok):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price reverses back toward midpoint or ADX weakens
            exit_signal = False
            midpoint = (r3_1d_aligned[i] + s3_1d_aligned[i]) / 2
            
            if position == 1:
                # Exit long when price reaches midpoint or ADX drops
                if close[i] >= midpoint or adx_1d_aligned[i] < 20:
                    exit_signal = True
            elif position == -1:
                # Exit short when price reaches midpoint or ADX drops
                if close[i] <= midpoint or adx_1d_aligned[i] < 20:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Camarilla_R3S3_Fade_12hVolumeSpike_1dADXTrend"
timeframe = "6h"
leverage = 1.0