#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # 1d data for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 12h data for Camarilla pivot calculation (current timeframe)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 12h high, low, close for Camarilla pivot calculation
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla pivot levels (based on previous 12h bar's OHLC)
    # Pivot = (H + L + C) / 3
    pivot_12h = (high_12h + low_12h + close_12h) / 3
    # Range = H - L
    range_12h = high_12h - low_12h
    # Resistance levels
    r1_12h = close_12h + (range_12h * 1.1 / 12)
    r2_12h = close_12h + (range_12h * 1.1 / 6)
    r3_12h = close_12h + (range_12h * 1.1 / 4)
    # Support levels
    s1_12h = close_12h - (range_12h * 1.1 / 12)
    s2_12h = close_12h - (range_12h * 1.1 / 6)
    s3_12h = close_12h - (range_12h * 1.1 / 4)
    
    # Align 1d EMA50 and 12h Camarilla levels to 12h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    pivot_12h_aligned = align_htf_to_ltf(prices, df_12h, pivot_12h)
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    r2_12h_aligned = align_htf_to_ltf(prices, df_12h, r2_12h)
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    s2_12h_aligned = align_htf_to_ltf(prices, df_12h, s2_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for EMA50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(pivot_12h_aligned[i]) or 
            np.isnan(r1_12h_aligned[i]) or np.isnan(s1_12h_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price above R1, above 1d EMA50, volume filter
            long_cond = (close[i] > r1_12h_aligned[i]) and (close[i] > ema_50_1d_aligned[i]) and volume_filter[i]
            # Short conditions: price below S1, below 1d EMA50, volume filter
            short_cond = (close[i] < s1_12h_aligned[i]) and (close[i] < ema_50_1d_aligned[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below pivot
            if close[i] < pivot_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above pivot
            if close[i] > pivot_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals