#!/usr/bin/env python3
"""
4h_Williams_Alligator_1dTrend_Filter
Hypothesis: Williams Alligator (Jaw/Teeth/Lips) identifies trend direction when aligned, 
combined with 1d EMA50 trend filter for higher timeframe confirmation. 
Enter when all three Alligator lines are aligned (bullish: Lips > Teeth > Jaw OR bearish: Lips < Teeth < Jaw) 
and price is in direction of 1d trend. Uses Williams Alligator's natural smoothing to reduce whipsaw. 
Target: 20-30 trades/year (80-120 total) to minimize fee drag while capturing sustained trends.
"""

name = "4h_Williams_Alligator_1dTrend_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(arr, period):
    """Smoothed Moving Average (SMMA) - Williams Alligator uses this"""
    if len(arr) < period:
        return np.full_like(arr, np.nan, dtype=float)
    result = np.full_like(arr, np.nan, dtype=float)
    # First value is simple average
    result[period-1] = np.mean(arr[:period])
    # Subsequent values: SMMA = (PREV_SMMA * (N-1) + CURRENT_CLOSE) / N
    for i in range(period, len(arr)):
        result[i] = (result[i-1] * (period - 1) + arr[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Williams Alligator on 4h timeframe
    # Jaw: Blue line - SMMA(13, 8)
    # Teeth: Red line - SMMA(8, 5) 
    # Lips: Green line - SMMA(5, 3)
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Shift the SMMA lines as per Williams Alligator specification
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    # Set rolled values to NaN
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # 1d EMA50 for trend filter
    ema50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema50_1d[49] = np.mean(close_1d[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema50_1d[i-1]
    
    # Align 1d EMA50 to 4h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for Alligator lines to stabilize
    
    for i in range(start_idx, n):
        if np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(ema50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams Alligator alignment signals
        bullish_aligned = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_aligned = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # 1d trend filter
        is_uptrend = close[i] > ema50_1d_aligned[i]
        is_downtrend = close[i] < ema50_1d_aligned[i]
        
        if position == 0:
            # Long: Alligator bullish alignment + 1d uptrend
            if bullish_aligned and is_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish alignment + 1d downtrend
            elif bearish_aligned and is_downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Alligator alignment breaks down or trend changes
            if not bullish_aligned or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Alligator alignment breaks down or trend changes
            if not bearish_aligned or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals