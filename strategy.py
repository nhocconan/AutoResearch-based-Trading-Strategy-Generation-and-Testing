#!/usr/bin/env python3
# 6h_WilliamsAlligator_ElderRay_1dTrend
# Hypothesis: Combine Williams Alligator (trend filter) from 1d chart with Elder Ray (Bull/Bear power) on 6h for entry timing.
# The Alligator (Jaw/Teeth/Lips) from 1d determines market regime - only trade when aligned.
# Elder Ray uses 13-period EMA: Bull Power = High - EMA13, Bear Power = EMA13 - Low.
# Enter long when Bull Power > 0 and rising, Bear Power < 0 in bullish Alligator alignment.
# Enter short when Bear Power > 0 and rising, Bull Power < 0 in bearish Alligator alignment.
# This combines trend-following with momentum to capture sustained moves while avoiding whipsaws.
# Designed for 6h timeframe with 1d Alligator filter to reduce trades and improve quality.
# Target: 15-30 trades/year to stay within optimal range for 6h.

name = "6h_WilliamsAlligator_ElderRay_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams Alligator on 1d: SMMA (Smoothed Moving Average) with specific periods
    jaw_period = 13  # Blue line
    teeth_period = 8  # Red line  
    lips_period = 5   # Green line
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < max(jaw_period, teeth_period, lips_period):
        return np.zeros(n)
    
    # Calculate SMMA (Smoothed Moving Average) - equivalent to RMA/Wilder's smoothing
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: smoothed
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw_1d = smma(df_1d['close'].values, jaw_period)
    teeth_1d = smma(df_1d['close'].values, teeth_period)
    lips_1d = smma(df_1d['close'].values, lips_period)
    
    # Align Alligator lines to 6h timeframe
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # Elder Ray on 6h: Bull Power and Bear Power using 13-period EMA
    ema13_period = 13
    ema13 = pd.Series(close).ewm(span=ema13_period, adjust=False, min_periods=ema13_period).mean().values
    
    bull_power = high - ema13  # High - EMA13
    bear_power = ema13 - low   # EMA13 - Low
    
    # Smooth the power indicators to reduce noise
    bull_power_smooth = pd.Series(bull_power).ewm(span=5, adjust=False, min_periods=5).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Determine Alligator alignment: 
    # Bullish: Lips > Teeth > Jaw (green > red > blue)
    # Bearish: Jaw > Teeth > Lips (blue > red > green)
    bullish_alignment = (lips_1d_aligned > teeth_1d_aligned) & (teeth_1d_aligned > jaw_1d_aligned)
    bearish_alignment = (jaw_1d_aligned > teeth_1d_aligned) & (teeth_1d_aligned > lips_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(jaw_period, teeth_period, lips_period, ema13_period) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(jaw_1d_aligned[i]) or np.isnan(teeth_1d_aligned[i]) or 
            np.isnan(lips_1d_aligned[i]) or np.isnan(bull_power_smooth[i]) or 
            np.isnan(bear_power_smooth[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: Bullish Alligator alignment AND Bull Power positive and rising
            if (bullish_alignment[i] and 
                bull_power_smooth[i] > 0 and 
                i > start_idx and bull_power_smooth[i] > bull_power_smooth[i-1]):
                signals[i] = 0.25
                position = 1
            # Short entry: Bearish Alligator alignment AND Bear Power positive and rising
            elif (bearish_alignment[i] and 
                  bear_power_smooth[i] > 0 and 
                  i > start_idx and bear_power_smooth[i] > bear_power_smooth[i-1]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Either Alligator turns bearish OR Bull Power becomes negative
            if not bullish_alignment[i] or bull_power_smooth[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Either Alligator turns bullish OR Bear Power becomes negative
            if not bearish_alignment[i] or bear_power_smooth[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals