#!/usr/bin/env python3
# 6h_1d_adaptive_range_breakout_v1
# Hypothesis: 6-hour breakouts above/below dynamic range levels (ATR-based) derived from daily price action.
# Uses dynamic range (daily ATR * multiplier) around daily midpoint as support/resistance.
# Works in both bull and bear markets as ranges adapt to volatility. Exit when price returns to daily midpoint.
# Includes volume confirmation to filter breakouts.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_adaptive_range_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for dynamic range calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for dynamic range
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    
    # ATR(14) using Wilder's smoothing (same as RMA)
    atr_1d = np.full_like(tr, np.nan, dtype=np.float64)
    if len(tr) >= 14:
        atr_1d[13] = np.mean(tr[:14])  # First ATR is simple average
        for i in range(14, len(tr)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Daily midpoint (average of high and low)
    midpoint_1d = (high_1d + low_1d) / 2.0
    
    # Dynamic range levels: midpoint ± (ATR * multiplier)
    multiplier = 1.5
    upper_range_1d = midpoint_1d + (atr_1d * multiplier)
    lower_range_1d = midpoint_1d - (atr_1d * multiplier)
    
    # Align 1d levels to 6h timeframe
    midpoint_aligned = align_htf_to_ltf(prices, df_1d, midpoint_1d)
    upper_range_aligned = align_htf_to_ltf(prices, df_1d, upper_range_1d)
    lower_range_aligned = align_htf_to_ltf(prices, df_1d, lower_range_1d)
    
    # Volume confirmation - 20 period average
    vol_ma_20 = np.full(n, np.nan)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if np.isnan(midpoint_aligned[i]) or np.isnan(upper_range_aligned[i]) or np.isnan(lower_range_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to or below daily midpoint
            if close[i] <= midpoint_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to or above daily midpoint
            if close[i] >= midpoint_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above upper range with volume confirmation
            if close[i] > upper_range_aligned[i] and volume[i] > vol_ma_20[i] * 1.5:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below lower range with volume confirmation
            elif close[i] < lower_range_aligned[i] and volume[i] > vol_ma_20[i] * 1.5:
                position = -1
                signals[i] = -0.25
    
    return signals