#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Alligator + Elder Ray Power confluence with weekly pivot filter.
Long when Alligator is bullish (jaw < teeth < lips) AND Bull Power > 0 AND price > weekly pivot.
Short when Alligator is bearish (jaw > teeth > lips) AND Bear Power < 0 AND price < weekly pivot.
Exit when Alligator reverses or power crosses zero.
Uses 1w HTF for pivot levels and 1d for Elder Ray/EMA13 to avoid whipsaws. Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 1w HTF for weekly pivot (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Weekly pivot: (weekly high + weekly low + weekly close) / 3
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Calculate 1d HTF for Elder Ray (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Elder Ray: Bull Power = high - EMA13, Bear Power = low - EMA13
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1d - ema_13_1d
    bear_power = low_1d - ema_13_1d
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Calculate 6h Williams Alligator (primary TF)
    # Jaw: 13-period SMMA smoothed 8 periods
    # Teeth: 8-period SMMA smoothed 5 periods  
    # Lips: 5-period SMMA smoothed 3 periods
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full_like(arr, np.nan)
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw_raw = smma(close, 13)
    teeth_raw = smma(close, 8)
    lips_raw = smma(close, 5)
    
    # Apply smoothing periods
    jaw = smma(jaw_raw, 8) if not np.all(np.isnan(jaw_raw)) else np.full(n, np.nan)
    teeth = smma(teeth_raw, 5) if not np.all(np.isnan(teeth_raw)) else np.full(n, np.nan)
    lips = smma(lips_raw, 3) if not np.all(np.isnan(lips_raw)) else np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 25)  # Allow for all smoothing
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        weekly_pivot_val = weekly_pivot_aligned[i]
        bull_val = bull_power_aligned[i]
        bear_val = bear_power_aligned[i]
        
        # Alligator conditions
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        
        alligator_bullish = jaw_val < teeth_val < lips_val
        alligator_bearish = jaw_val > teeth_val > lips_val
        
        if position == 0:
            # Long: Alligator bullish AND Bull Power > 0 AND price > weekly pivot
            if alligator_bullish and bull_val > 0 and price > weekly_pivot_val:
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish AND Bear Power < 0 AND price < weekly pivot
            elif alligator_bearish and bear_val < 0 and price < weekly_pivot_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Alligator turns bearish OR Bull Power <= 0
                if not alligator_bullish or bull_val <= 0:
                    exit_signal = True
            elif position == -1:
                # Short exit: Alligator turns bullish OR Bear Power >= 0
                if not alligator_bearish or bear_val >= 0:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsAlligator_ElderRay_Power_WeeklyPivot"
timeframe = "6h"
leverage = 1.0