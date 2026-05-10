#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_Plus_1d_Volume_Spike
Hypothesis: Use 12h price touching Camarilla R3/S3 levels from prior day, filtered by daily volume spike (>2x 20-period average) and 1d EMA34 trend. Camarilla levels provide high-probability reversal points; volume spike confirms participation; EMA34 filter ensures trades align with daily trend. Designed for 12-25 trades/year per symbol, works in bull/bear via trend filter.
"""

name = "12h_Camarilla_Pivot_Plus_1d_Volume_Spike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior day (HLC of previous day)
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    # Camarilla calculations
    range_prev = high_prev - low_prev
    camarilla_pivot = (high_prev + low_prev + close_prev) / 3.0
    camarilla_r3 = camarilla_pivot + (range_prev * 1.1 / 4)
    camarilla_s3 = camarilla_pivot - (range_prev * 1.1 / 4)
    
    # Align Camarilla levels to 12h timeframe (prior day's levels available at 00:00 UTC daily)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Daily trend filter: EMA34
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Get 12h price and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 2x 20-period EMA (strong participation)
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > vol_ema20 * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need daily Camarilla (1 day lag), EMA34 (34), volume EMA (20)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema34_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price touches or crosses S3 from below AND daily uptrend AND volume spike
            if low[i] <= camarilla_s3_aligned[i] and close[i] > camarilla_s3_aligned[i] and close[i] > ema34_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price touches or crosses R3 from above AND daily downtrend AND volume spike
            elif high[i] >= camarilla_r3_aligned[i] and close[i] < camarilla_r3_aligned[i] and close[i] < ema34_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price touches or crosses R3 OR daily trend turns down
            if high[i] >= camarilla_r3_aligned[i] or close[i] < ema34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price touches or crosses S3 OR daily trend turns up
            if low[i] <= camarilla_s3_aligned[i] or close[i] > ema34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals