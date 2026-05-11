#!/usr/bin/env python3
"""
6h_Alligator_Trend_Filtered_Pullback
Hypothesis: Use Williams Alligator (13/8/5 SMAs) on 1d to define trend, then enter pullbacks
on 6h when price touches the Alligator's teeth (8-period SMA) in direction of trend.
The Alligator acts as a dynamic support/resistance in trending markets, providing
high-probability pullback entries. Works in both bull and bear markets by following
the higher timeframe trend, avoiding counter-trend whipsaws. Low frequency due to
strict pullback requirement and trend filter.
"""

name = "6h_Alligator_Trend_Filtered_Pullback"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for Alligator and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # 6h OHLCV
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # --- 1d Williams Alligator ---
    # Jaw: 13-period SMMA, shifted 8 bars
    # Teeth: 8-period SMMA, shifted 5 bars  
    # Lips: 5-period SMMA, shifted 3 bars
    close_1d = df_1d['close'].values
    
    def smma(arr, period):
        """Smoothed Moving Average (SMMA) - same as RMA/Wilder's"""
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        result = np.full_like(arr, np.nan, dtype=float)
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: (prev * (period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw_raw = smma(close_1d, 13)
    teeth_raw = smma(close_1d, 8)
    lips_raw = smma(close_1d, 5)
    
    # Shift as per Alligator definition
    jaw = np.roll(jaw_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    
    # Align to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Trend direction: price > teeth = uptrend, price < teeth = downtrend
    trend_up = close > teeth_aligned
    trend_down = close < teeth_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Entry conditions: Pullback to teeth in direction of trend
        # Long: price touches/slightly above teeth during uptrend
        # Short: price touches/slightly below teeth during downtrend
        # Using 0.1% buffer to avoid whipsaws
        buffer = teeth_aligned[i] * 0.001
        long_entry = (close[i] >= teeth_aligned[i] - buffer) and (close[i] <= teeth_aligned[i] + buffer) and trend_up[i]
        short_entry = (close[i] >= teeth_aligned[i] - buffer) and (close[i] <= teeth_aligned[i] + buffer) and trend_down[i]
        
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        else:
            # Exit conditions: Price moves beyond lips (Alligator waking up)
            if position == 1:
                # Exit if price crosses below lips (trend weakening)
                if close[i] < lips_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit if price crosses above lips (trend weakening)
                if close[i] > lips_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals