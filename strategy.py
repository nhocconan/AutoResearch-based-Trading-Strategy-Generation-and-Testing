# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
12h_1d_alligator_trend
Uses Williams Alligator on daily timeframe to identify trend direction.
Enters on 12h when price crosses above/below Alligator teeth with volume confirmation.
Exits when price crosses back below/above teeth or momentum weakens.
Williams Alligator: Jaw (13-period SMMA, 8 offset), Teeth (8-period SMMA, 5 offset), Lips (5-period SMMA, 3 offset).
Designed for low trade frequency (target: 15-30 trades/year) to minimize fee drift.
Works in trending markets by following the Alligator's alignment.
"""

name = "12h_1d_alligator_trend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(series, period):
    """Smoothed Moving Average (SMMA)"""
    if len(series) < period:
        return np.full_like(series, np.nan, dtype=float)
    result = np.full_like(series, np.nan, dtype=float)
    # First value is simple average
    result[period-1] = np.mean(series[:period])
    # Subsequent values: SMMA = (prev_SMMA * (period-1) + current_value) / period
    for i in range(period, len(series)):
        result[i] = (result[i-1] * (period-1) + series[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Alligator calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Williams Alligator components
    # Jaw: 13-period SMMA, 8 bars offset
    jaw_raw = smma(close_1d, 13)
    jaw = np.roll(jaw_raw, 8)  # shift forward by 8 bars
    # Teeth: 8-period SMMA, 5 bars offset
    teeth_raw = smma(close_1d, 8)
    teeth = np.roll(teeth_raw, 5)  # shift forward by 5 bars
    # Lips: 5-period SMMA, 3 bars offset
    lips_raw = smma(close_1d, 5)
    lips = np.roll(lips_raw, 3)  # shift forward by 3 bars
    
    # Align Alligator lines to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Volume confirmation on 12h: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: price crosses above teeth with lips above jaws (bullish alignment) and volume
        if (close[i] > teeth_aligned[i] and lips_aligned[i] > jaw_aligned[i] and 
            vol_confirm[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: price crosses below teeth with lips below jaws (bearish alignment) and volume
        elif (close[i] < teeth_aligned[i] and lips_aligned[i] < jaw_aligned[i] and 
              vol_confirm[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit conditions: price crosses back below/above teeth
        elif position == 1 and close[i] < teeth_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > teeth_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals