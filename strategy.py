#!/usr/bin/env python3
# Hypothesis: 6h timeframe with weekly pivot levels as key support/resistance and daily trend alignment.
# Strategy uses weekly pivot points (R4/R3/S3/S4 levels) from the prior week to identify key levels.
# Enters long when price breaks above weekly R3 with daily EMA50 alignment and volume confirmation.
# Enters short when price breaks below weekly S3 with daily EMA50 alignment and volume confirmation.
# Exits when price returns to weekly pivot (PP) or reverses at opposite S3/R3 levels.
# Uses volume spike (volume > 1.5x 20-period average) to confirm breakouts.
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25.

name = "6h_WeeklyPivot_R3S3_Breakout_DailyTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate weekly pivot points from prior week
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Use prior week's OHLC for pivot calculation (avoid look-ahead)
    # Shift by 1 week to use completed week data
    high_1w = df_1w['high'].shift(1)
    low_1w = df_1w['low'].shift(1)
    close_1w = df_1w['close'].shift(1)
    
    # Calculate pivot points
    pivot_p = (high_1w + low_1w + close_1w) / 3
    r1 = 2 * pivot_p - low_1w
    s1 = 2 * pivot_p - high_1w
    r2 = pivot_p + (high_1w - low_1w)
    s2 = pivot_p - (high_1w - low_1w)
    r3 = high_1w + 2 * (pivot_p - low_1w)
    s3 = low_1w - 2 * (high_1w - pivot_p)
    r4 = pivot_p + 3 * (high_1w - low_1w)
    s4 = pivot_p - 3 * (high_1w - low_1w)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_p_aligned = align_htf_to_ltf(prices, df_1w, pivot_p.values)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3.values)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3.values)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4.values)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4.values)
    
    # Daily EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50 = df_1d['close'].ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume spike detection (volume > 1.5x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(pivot_p_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above weekly R3 + daily EMA50 uptrend + volume spike
            if (close[i] > r3_aligned[i] and 
                close[i] > ema_50_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below weekly S3 + daily EMA50 downtrend + volume spike
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema_50_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to weekly pivot or reverses at S3
            if (close[i] <= pivot_p_aligned[i] or close[i] <= s3_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to weekly pivot or reverses at R3
            if (close[i] >= pivot_p_aligned[i] or close[i] >= r3_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals