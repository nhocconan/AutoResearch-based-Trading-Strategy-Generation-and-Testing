#!/usr/bin/env python3
# 6H_1D_Alligator_Trend_Confirm
# Hypothesis: Use Williams Alligator on 1d to determine trend (jaws-teeth-lips alignment) and enter on 6h breakouts in trend direction.
# Alligator provides smoothed trend filtering, reducing whipsaws in choppy markets. Works in both bull/bear by following the trend.
# Target: 15-30 trades/year per symbol (60-120 total over 4 years).

name = "6H_1D_Alligator_Trend_Confirm"
timeframe = "6h"
leverage = 1.0

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
    
    # Get 1d data for Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Alligator: three SMAs with different periods and shifts
    # Jaw (Blue): 13-period SMMA, shifted 8 bars forward
    # Teeth (Red): 8-period SMMA, shifted 5 bars forward  
    # Lips (Green): 5-period SMMA, shifted 3 bars forward
    
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full_like(arr, np.nan, dtype=float)
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: (prev*(period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    # Calculate Alligator components
    jaw = smma(close_1d, 13)
    teeth = smma(close_1d, 8)
    lips = smma(close_1d, 5)
    
    # Apply shifts (Jaw: +8, Teeth: +5, Lips: +3)
    jaw_shifted = np.full_like(jaw, np.nan)
    teeth_shifted = np.full_like(teeth, np.nan)
    lips_shifted = np.full_like(lips, np.nan)
    
    if len(jaw) > 8:
        jaw_shifted[8:] = jaw[:-8]
    if len(teeth) > 5:
        teeth_shifted[5:] = teeth[:-5]
    if len(lips) > 3:
        lips_shifted[3:] = lips[:-3]
    
    # Trend determination: aligned = bullish, reversed = bearish, intertwined = chop
    jaw_val = jaw_shifted
    teeth_val = teeth_shifted
    lips_val = lips_shifted
    
    bullish_alignment = (lips_val > teeth_val) & (teeth_val > jaw_val)
    bearish_alignment = (jaw_val > teeth_val) & (teeth_val > lips_val)
    
    # Align Alligator components to 6h
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_val)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_val)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_val)
    bullish_aligned = align_htf_to_ltf(prices, df_1d, bullish_alignment.astype(float))
    bearish_aligned = align_htf_to_ltf(prices, df_1d, bearish_alignment.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bullish_aligned[i]) or np.isnan(bearish_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bullish = bullish_aligned[i] > 0.5
        bearish = bearish_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: bullish alignment + price above teeth (Alligator's "mouth open" up)
            if bullish and close[i] > teeth_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish alignment + price below teeth (Alligator's "mouth open" down)
            elif bearish and close[i] < teeth_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish alignment or price falls below jaws
            if bearish or close[i] < jaw_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish alignment or price rises above jaws
            if bullish or close[i] > jaw_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals