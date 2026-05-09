#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Camarilla_R3_S3_Breakout_WeeklyTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla levels and volume context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels from previous day's range
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values (already closed)
    high_1d_prev = np.roll(high_1d, 1)
    low_1d_prev = np.roll(low_1d, 1)
    close_1d_prev = np.roll(close_1d, 1)
    high_1d_prev[0] = np.nan
    low_1d_prev[0] = np.nan
    close_1d_prev[0] = np.nan
    
    pp_1d = (high_1d_prev + low_1d_prev + close_1d_prev) / 3
    range_1d = high_1d_prev - low_1d_prev
    r3_1d = close_1d_prev + range_1d * 1.1 * 3 / 4
    s3_1d = close_1d_prev - range_1d * 1.1 * 3 / 4
    r4_1d = close_1d_prev + range_1d * 1.1 * 1 / 2
    s4_1d = close_1d_prev - range_1d * 1.1 * 1 / 2
    
    # Align daily levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Weekly trend filter: EMA(21) on weekly close
    close_1w = df_1w['close'].values
    ema21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    # Volume confirmation: current volume > 1.8x 24-period average (~6 days for 6h)
    vol_series = pd.Series(volume)
    vol_ma24 = vol_series.rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema21_1w_aligned[i]) or np.isnan(pp_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(vol_ma24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.8 * vol_ma24[i]
        
        if position == 0:
            # Long: Close breaks above R3 with volume spike and above weekly EMA trend
            if close[i] > r3_aligned[i] and vol_ok and close[i] > ema21_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below S3 with volume spike and below weekly EMA trend
            elif close[i] < s3_aligned[i] and vol_ok and close[i] < ema21_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses back below R4 (take profit) or below S3 (stop)
            if close[i] < r4_aligned[i] or close[i] < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses back above S4 (take profit) or above R3 (stop)
            if close[i] > s4_aligned[i] or close[i] > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals