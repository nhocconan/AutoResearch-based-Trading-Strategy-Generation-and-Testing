#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Breakout_1wTrend_Volume
Hypothesis: Use 6h Camarilla R3/S3 breakout with weekly trend filter and volume confirmation.
Weekly trend from 1w close vs 20-period SMA determines market direction (bull/bear).
Camarilla levels from 1d provide precise entry/exit points. Volume filter ensures breakout validity.
Designed for 12-30 trades/year per symbol, works in bull/bear via trend filter.
"""

name = "6h_Camarilla_R3_S3_Breakout_1wTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly 20-period SMA for trend filter
    close_1w = df_1w['close'].values
    sma_20_1w = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    sma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_20_1w)
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla levels using previous day
    high_d = df_1d['high'].shift(1).values
    low_d = df_1d['low'].shift(1).values
    close_d = df_1d['close'].shift(1).values
    
    # Camarilla levels
    camarilla_base = (high_d + low_d + close_d) / 3.0
    camarilla_range = high_d - low_d
    r3 = camarilla_base + camarilla_range * 1.1 / 4
    s3 = camarilla_base - camarilla_range * 1.1 / 4
    r4 = camarilla_base + camarilla_range * 1.1 / 2
    s4 = camarilla_base - camarilla_range * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe
    camarilla_base_aligned = align_htf_to_ltf(prices, df_1d, camarilla_base)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Get 6h price and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.3x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly SMA (20) and volume EMA (20)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(sma_20_1w_aligned[i]) or 
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or
            np.isnan(s4_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend: close vs 20-period SMA
        bullish_trend = close[i] > sma_20_1w_aligned[i]
        bearish_trend = close[i] < sma_20_1w_aligned[i]
        
        if position == 0:
            # Long: bullish weekly trend AND price breaks above R3 with volume
            if bullish_trend and high[i] > r3_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish weekly trend AND price breaks below S3 with volume
            elif bearish_trend and low[i] < s3_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S3 OR weekly trend turns bearish
            if low[i] < s3_aligned[i] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R3 OR weekly trend turns bullish
            if high[i] > r3_aligned[i] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals