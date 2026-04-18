#!/usr/bin/env python3
"""
4h_1d_TRIX_Volume_Spike_Trend
Hypothesis: Uses TRIX (triple smoothed EMA) on 1d for trend direction and TRIX on 4h for momentum confirmation.
Adds volume spike filter to ensure momentum and avoid chop. Designed for low trade frequency (~20-30/year) with strong trend capture.
Works in bull (rides trends) and bear (avoids false signals via volume/spike filter).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(arr, period):
    """Calculate EMA with proper handling of NaN"""
    n = len(arr)
    result = np.full(n, np.nan)
    if n < period:
        return result
    k = 2 / (period + 1)
    # Initialize with SMA
    result[period-1] = np.mean(arr[:period])
    for i in range(period, n):
        result[i] = arr[i] * k + result[i-1] * (1 - k)
    return result

def trix(arr, period):
    """Calculate TRIX: triple EMA then percent change"""
    e1 = ema(arr, period)
    e2 = ema(e1, period)
    e3 = ema(e2, period)
    # Percent change: (current - previous) / previous * 100
    result = np.full_like(e3, np.nan)
    result[1:] = (e3[1:] - e3[:-1]) / e3[:-1] * 100
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for TRIX trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate TRIX on 1d (trend filter)
    trix_1d = trix(close_1d, 12)
    trix_1d_aligned = align_htf_to_ltf(prices, df_1d, trix_1d)
    
    # Calculate TRIX on 4h (momentum)
    trix_4h = trix(close, 12)
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Warmup for TRIX calculations
    
    for i in range(start_idx, n):
        if (np.isnan(trix_1d_aligned[i]) or np.isnan(trix_4h[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: TRIX 1d positive (uptrend), TRIX 4h turning up, volume spike
            if trix_1d_aligned[i] > 0 and trix_4h[i] > trix_4h[i-1] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: TRIX 1d negative (downtrend), TRIX 4h turning down, volume spike
            elif trix_1d_aligned[i] < 0 and trix_4h[i] < trix_4h[i-1] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: TRIX 1d turns negative or TRIX 4h turns down
            if trix_1d_aligned[i] < 0 or trix_4h[i] < trix_4h[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: TRIX 1d turns positive or TRIX 4h turns up
            if trix_1d_aligned[i] > 0 or trix_4h[i] > trix_4h[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_TRIX_Volume_Spike_Trend"
timeframe = "4h"
leverage = 1.0