#!/usr/bin/env python3
"""
4h_WilliamsAlligator_Trend_Filter
Hypothesis: Use Williams Alligator (3 SMAs: Jaw=13, Teeth=8, Lips=5) on 1d timeframe to determine trend direction, with 4h price crossing above/below the Lips line as entry signal. Only take longs in uptrend (Lips > Teeth > Jaw) and shorts in downtrend (Lips < Teeth < Jaw). This avoids whipsaw in ranging markets and captures strong trends in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === Williams Alligator on 1d ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Jaw (13-period SMMA), Teeth (8-period), Lips (5-period)
    def smma(arr, period):
        # Smoothed Moving Average: similar to EMA but with alpha = 1/period
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (prev * (period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close_1d, 13)
    teeth = smma(close_1d, 8)
    lips = smma(close_1d, 5)
    
    # Trend condition: Lips > Teeth > Jaw for uptrend, Lips < Teeth < Jaw for downtrend
    jaw_up = lips > teeth
    teeth_up = teeth > jaw
    uptrend = jaw_up & teeth_up  # Lips > Teeth > Jaw
    
    jaw_down = lips < teeth
    teeth_down = teeth < jaw
    downtrend = jaw_down & teeth_down  # Lips < Teeth < Jaw
    
    # Align to 4h
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    uptrend_aligned = align_htf_to_ltf(prices, df_1d, uptrend.astype(float))
    downtrend_aligned = align_htf_to_ltf(prices, df_1d, downtrend.astype(float))
    
    signals = np.zeros(n)
    
    # Warmup: enough for SMMA calculation
    warmup = 20  # Covers longest period (13) plus some buffer
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(lips_aligned[i]) or np.isnan(uptrend_aligned[i]) or 
            np.isnan(downtrend_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: Price crosses above Lips AND uptrend
            if close[i] > lips_aligned[i] and uptrend_aligned[i] > 0.5:
                signals[i] = 0.25
                position = 1
                continue
            # Short: Price crosses below Lips AND downtrend
            elif close[i] < lips_aligned[i] and downtrend_aligned[i] > 0.5:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: reverse signal
        elif position == 1:
            # Exit when price crosses below Lips (trend change)
            if close[i] < lips_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price crosses above Lips (trend change)
            if close[i] > lips_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_Trend_Filter"
timeframe = "4h"
leverage = 1.0