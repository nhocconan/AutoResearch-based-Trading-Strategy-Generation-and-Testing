#!/usr/bin/env python3
"""
12h Camarilla Pivot Breakout + Volume Spike + Daily Trend Filter
Hypothesis: Camarilla pivot levels (from daily) act as strong support/resistance. 
Breakouts above R3 or below S3 with volume confirmation and aligned daily trend 
capture institutional moves. Works in bull (breakouts up) and bear (breakouts down) 
markets by following the daily trend direction. Low trade frequency due to 
requirement of strong breaks beyond inner S1/R1 levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def calculate_camarilla_pivot(high, low, close):
    """Calculate Camarilla pivot levels for the day"""
    # Typical price
    typical_price = (high + low + close) / 3
    # Range
    range_val = high - low
    
    # Camarilla levels
    pivot = typical_price
    r1 = close + (range_val * 1.1 / 12)
    r2 = close + (range_val * 1.1 / 6)
    r3 = close + (range_val * 1.1 / 4)
    s1 = close - (range_val * 1.1 / 12)
    s2 = close - (range_val * 1.1 / 6)
    s3 = close - (range_val * 1.1 / 4)
    
    return pivot, r1, r2, r3, s1, s2, s3

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data for pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Initialize arrays for pivot levels
    r3_1d = np.full_like(close_1d, np.nan)
    s3_1d = np.full_like(close_1d, np.nan)
    r1_1d = np.full_like(close_1d, np.nan)
    s1_1d = np.full_like(close_1d, np.nan)
    trend_up = np.zeros_like(close_1d, dtype=bool)
    
    # Calculate Camarilla levels for each day
    for i in range(len(close_1d)):
        _, r1, r2, r3, s1, s2, s3 = calculate_camarilla_pivot(high_1d[i], low_1d[i], close_1d[i])
        r1_1d[i] = r1
        r2_1d[i] = r2
        r3_1d[i] = r3
        s1_1d[i] = s1
        s2_1d[i] = s2
        s3_1d[i] = s3
        # Simple trend: close above/below previous day's close
        if i > 0:
            trend_up[i] = close_1d[i] > close_1d[i-1]
        else:
            trend_up[i] = True  # Default for first day
    
    # Align daily levels to 12h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    trend_aligned = align_htf_to_ltf(prices, df_1d, trend_up.astype(float))
    
    # 12h price and volume
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume spike: current volume > 1.5x 24-period average (2 days of 12h data)
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 24:
            vol_ma[i] = np.mean(volume[max(0, i-23):i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-23:i+1])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Warmup
    
    for i in range(start_idx, n):
        if np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        r3 = r3_1d_aligned[i]
        s3 = s3_1d_aligned[i]
        r1 = r1_1d_aligned[i]
        s1 = s1_1d_aligned[i]
        trend = trend_aligned[i] > 0.5  # Boolean: True for uptrend
        vol_ok = vol_spike[i]
        
        if position == 0:
            # Enter long: price breaks above R3 (strong resistance) + volume spike + daily uptrend
            if (close[i] > r3 and 
                vol_ok and 
                trend):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S3 (strong support) + volume spike + daily downtrend
            elif (close[i] < s3 and 
                  vol_ok and 
                  not trend):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price drops back below R1 (inner resistance) or trend reverses
            if close[i] < r1 or trend_aligned[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above S1 (inner support) or trend reverses
            if close[i] > s1 or trend_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_Pivot_Breakout_VolumeSpike_Trend"
timeframe = "12h"
leverage = 1.0