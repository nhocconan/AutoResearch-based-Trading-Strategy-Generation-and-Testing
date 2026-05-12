#!/usr/bin/env python3
# 1D_WILLIAMS_ALLIGATOR_TREND_FOLLOW
# Hypothesis: Williams Alligator (Jaw=TEETH=LIPS SMMA) identifies trend absence/presence.
# In trending markets (JAW > TEETH > LIPS for uptrend, reverse for downtrend), follow trend with 20% position.
# Uses 1-day SMMA for Alligator lines, aligned to 1d timeframe. Avoids chop, captures sustained moves.
# Works in both bull and bear: only trades when clear trend present, avoids whipsaws.
# Target: 10-20 trades/year on 1d timeframe.

name = "1D_WILLIAMS_ALLIGATOR_TREND_FOLLOW"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(arr, period):
    """Smoothed Moving Average (SMMA) - same as Wilder's smoothing"""
    if len(arr) < period:
        return np.full_like(arr, np.nan, dtype=float)
    res = np.full_like(arr, np.nan, dtype=float)
    # First value is simple average
    res[period-1] = np.mean(arr[:period])
    # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
    for i in range(period, len(arr)):
        res[i] = (res[i-1] * (period-1) + arr[i]) / period
    return res

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Daily data for Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:  # Need enough for SMMA(13)
        return np.zeros(n)
    
    # Williams Alligator parameters (standard: 13,8,5)
    jaw_period = 13   # Blue line
    teeth_period = 8  # Red line  
    lips_period = 5   # Green line
    
    # Calculate SMMA for each line
    jaw = smma(df_1d['close'].values, jaw_period)
    teeth = smma(df_1d['close'].values, teeth_period)
    lips = smma(df_1d['close'].values, lips_period)
    
    # Align to 1d timeframe (no additional delay needed for SMMA)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(jaw_period, teeth_period, lips_period)  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: JAW > TEETH > LIPS (bullish alignment)
            if (jaw_aligned[i] > teeth_aligned[i] and 
                teeth_aligned[i] > lips_aligned[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: JAW < TEETH < LIPS (bearish alignment)
            elif (jaw_aligned[i] < teeth_aligned[i] and 
                  teeth_aligned[i] < lips_aligned[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Trend weakening (JAW <= TEETH or TEETH <= LIPS)
            if (jaw_aligned[i] <= teeth_aligned[i] or 
                teeth_aligned[i] <= lips_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Trend weakening (JAW >= TEETH or TEETH >= LIPS)
            if (jaw_aligned[i] >= teeth_aligned[i] or 
                teeth_aligned[i] >= lips_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals