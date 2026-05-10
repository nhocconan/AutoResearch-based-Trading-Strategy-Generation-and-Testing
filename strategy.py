#!/usr/bin/env python3
"""
1d_1D_Camarilla_R3_S3_Breakout_1wTrend_Volume
Hypothesis: Daily Camarilla R3/S3 breakout filtered by weekly trend (price above/below weekly pivot) and volume confirmation.
Designed for 8-15 trades/year per symbol, works in bull/bear via weekly trend filter.
"""

name = "1d_1D_Camarilla_R3_S3_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week)
    high_wk = df_1w['high'].shift(1).values
    low_wk = df_1w['low'].shift(1).values
    close_wk = df_1w['close'].shift(1).values
    
    pivot_wk = (high_wk + low_wk + close_wk) / 3.0
    
    # Align weekly pivot to daily timeframe
    pivot_wk_aligned = align_htf_to_ltf(prices, df_1w, pivot_wk)
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels (using prior day)
    high_d = df_1d['high'].shift(1).values
    low_d = df_1d['low'].shift(1).values
    close_d = df_1d['close'].shift(1).values
    
    # Camarilla R3 and S3 levels
    r3 = close_d + (high_d - low_d) * 1.1 / 4
    s3 = close_d - (high_d - low_d) * 1.1 / 4
    
    # Align Camarilla levels to daily timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Get daily price and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.5x 20-day EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly pivot and daily Camarilla
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(pivot_wk_aligned[i]) or 
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend: price vs weekly pivot
        bullish_trend = close[i] > pivot_wk_aligned[i]
        bearish_trend = close[i] < pivot_wk_aligned[i]
        
        if position == 0:
            # Long: bullish weekly trend AND price breaks above daily R3 with volume
            if bullish_trend and high[i] > r3_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish weekly trend AND price breaks below daily S3 with volume
            elif bearish_trend and low[i] < s3_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below daily S3 OR weekly trend turns bearish
            if low[i] < s3_aligned[i] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above daily R3 OR weekly trend turns bullish
            if high[i] > r3_aligned[i] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals