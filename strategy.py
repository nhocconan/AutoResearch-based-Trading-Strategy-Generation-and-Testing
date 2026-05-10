#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
Hypothesis: Use Camarilla R3/S3 levels from 1d for breakout entries in the direction of 1d EMA34 trend, filtered by volume spike.
Camarilla levels provide institutional support/resistance that work in both bull and bear markets via trend filter.
Designed for ~25-35 trades/year on 4h timeframe to avoid excessive fee drag.
"""

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla levels (using prior day)
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    # Camarilla equations
    range_prev = high_prev - low_prev
    camarilla_r3 = close_prev + range_prev * 1.1 / 2
    camarilla_s3 = close_prev - range_prev * 1.1 / 2
    
    # Align daily Camarilla levels to 4h timeframe (wait for daily bar to close)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Daily EMA34 for trend filter
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Get 4h price and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 2.0x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need daily Camarilla (needs 1 day), EMA34 (34 bars), volume EMA (20)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or
            np.isnan(ema_34_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: above EMA34 (uptrend) AND price breaks above Camarilla R3 with volume spike
            if close[i] > ema_34_aligned[i] and high[i] > r3_aligned[i] and volume_filter[i]:
                signals[i] = 0.30
                position = 1
            # Short: below EMA34 (downtrend) AND price breaks below Camarilla S3 with volume spike
            elif close[i] < ema_34_aligned[i] and low[i] < s3_aligned[i] and volume_filter[i]:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long exit: price breaks below Camarilla pivot (central level) OR trend turns bearish
            # Calculate Camarilla pivot for exit: (H+L+2*C)/4
            camarilla_pivot = (high_prev + low_prev + 2 * close_prev) / 4
            pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
            if low[i] < pivot_aligned[i] or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short exit: price breaks above Camarilla pivot OR trend turns bullish
            camarilla_pivot = (high_prev + low_prev + 2 * close_prev) / 4
            pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
            if high[i] > pivot_aligned[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals