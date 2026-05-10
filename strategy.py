#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_Volume
Hypothesis: Use daily Camarilla R3/S3 levels for 4h breakout entries in direction of 1d EMA34 trend, with volume confirmation. 
Camarilla levels are institutional pivot points that work in both bull and bear markets via trend filter.
Target: 20-50 trades/year (80-200 total over 4 years).
"""

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data for Camarilla and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels (using prior day)
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    range_prev = high_prev - low_prev
    
    # Camarilla formulas
    camarilla_p = (high_prev + low_prev + close_prev) / 3
    camarilla_r3 = camarilla_p + 1.1 * range_prev / 2
    camarilla_s3 = camarilla_p - 1.1 * range_prev / 2
    
    # Align daily Camarilla levels to 4h timeframe (wait for daily bar to close)
    camarilla_p_aligned = align_htf_to_ltf(prices, df_1d, camarilla_p)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Daily EMA34 for trend filter
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Get 4h price and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need daily Camarilla (needs 1 day), EMA34 (34 bars), volume EMA (20)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(camarilla_p_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_34_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: above EMA34 (uptrend) AND price breaks above Camarilla R3 with volume
            if close[i] > ema_34_aligned[i] and high[i] > camarilla_r3_aligned[i] and volume_filter[i]:
                signals[i] = 0.30
                position = 1
            # Short: below EMA34 (downtrend) AND price breaks below Camarilla S3 with volume
            elif close[i] < ema_34_aligned[i] and low[i] < camarilla_s3_aligned[i] and volume_filter[i]:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long exit: price breaks below camarilla P OR trend turns bearish
            if low[i] < camarilla_p_aligned[i] or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short exit: price breaks above camarilla P OR trend turns bullish
            if high[i] > camarilla_p_aligned[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals