#!/usr/bin/env python3
# 4h_PivotBreakout_R3S3_1dTrend_Volume
# Hypothesis: 4h chart strategy using daily pivot-based R3/S3 breakouts filtered by 1d EMA50 trend and volume confirmation (1.5x average volume).
# Daily R3/S3 act as strong support/resistance with high probability of reversal or breakout.
# 1d EMA50 provides trend filter to avoid counter-trend trades. Volume confirms breakout validity.
# Designed to work in both bull and bear markets by filtering with trend and requiring volume confirmation.
# Target: 20-40 trades/year per symbol to minimize fee drag while maintaining edge.

timeframe = "4h"
name = "4h_PivotBreakout_R3S3_1dTrend_Volume"
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
    
    # Get daily data for pivot points (R3, S3) and trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate daily pivot points: R3, S3
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_r3 = close_1d + 1.1 * (high_1d - low_1d)
    pivot_s3 = close_1d - 1.1 * (high_1d - low_1d)
    
    pivot_r3_aligned = align_htf_to_ltf(prices, df_1d, pivot_r3)
    pivot_s3_aligned = align_htf_to_ltf(prices, df_1d, pivot_s3)
    
    # Volume spike detection: 1.5x average volume (6-period = 1 day on 4h chart)
    vol_ma = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 6)  # Ensure we have EMA50 and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(pivot_r3_aligned[i]) or 
            np.isnan(pivot_s3_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above daily R3 with volume, and 1d trend is bullish (price > EMA50)
            if (high[i] > pivot_r3_aligned[i] and 
                volume[i] > 1.5 * vol_ma[i] and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below daily S3 with volume, and 1d trend is bearish (price < EMA50)
            elif (low[i] < pivot_s3_aligned[i] and 
                  volume[i] > 1.5 * vol_ma[i] and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below daily S3 (reversal signal)
            if low[i] < pivot_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above daily R3 (reversal signal)
            if high[i] > pivot_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals