#!/usr/bin/env python3
# Hypothesis: 6h Williams Alligator + Elder Ray (Bull/Bear Power) combination
# Long when: Price > Alligator Jaw (13-period smoothed median), Bull Power > 0 and rising, Bear Power < 0
# Short when: Price < Alligator Jaw, Bear Power < 0 and falling, Bull Power > 0
# Exit when: Price crosses back below/above Alligator Teeth (8-period smoothed median)
# Uses Alligator for trend direction and Elder Ray for power confirmation
# Designed to capture trend continuation with momentum filters
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25

name = "6h_Alligator_ElderRay_PowerTrend"
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
    
    # Calculate 1d Williams Alligator components
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Alligator: Jaw (13), Teeth (8), Lips (5) - all smoothed medians
    # Using median approximation via rolling median for simplicity
    median_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    
    jaw = median_price.rolling(window=13, min_periods=13).median()
    teeth = median_price.rolling(window=8, min_periods=8).median()
    lips = median_price.rolling(window=5, min_periods=5).median()
    
    # Smooth with SMMA (smoothed moving average) approximation
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (PERIOD-1) + CURRENT) / PERIOD
        for i in range(period, len(arr)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw_smma = smma(jaw.values, 13)
    teeth_smma = smma(teeth.values, 8)
    lips_smma = smma(lips.values, 5)
    
    # Align Alligator lines to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_smma)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_smma)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_smma)
    
    # Calculate Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = median_price.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    bull_power = df_1d['high'].values - ema13
    bear_power = df_1d['low'].values - ema13
    
    # Align Elder Ray components
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need enough data for indicator calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Price > Jaw, Bull Power > 0 and rising, Bear Power < 0
            if (close[i] > jaw_aligned[i] and 
                bull_power_aligned[i] > 0 and 
                bull_power_aligned[i] > bull_power_aligned[i-1] and
                bear_power_aligned[i] < 0):
                signals[i] = 0.25
                position = 1
            # Enter short: Price < Jaw, Bear Power < 0 and falling, Bull Power > 0
            elif (close[i] < jaw_aligned[i] and 
                  bear_power_aligned[i] < 0 and 
                  bear_power_aligned[i] < bear_power_aligned[i-1] and
                  bull_power_aligned[i] > 0):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below Teeth (8-period smoothed median)
            if close[i] < teeth_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above Teeth
            if close[i] > teeth_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals