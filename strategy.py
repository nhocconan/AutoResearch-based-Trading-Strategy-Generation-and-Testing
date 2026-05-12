#!/usr/bin/env python3
# 6h_Williams_Alligator_ADX_Trend_1d
# Hypothesis: Williams Alligator (JAW/TEETH/LIPS) on 6h combined with ADX on 1d for trend strength.
# The Alligator identifies trend direction when lines are aligned and non-intertwined.
# ADX > 25 confirms sufficient trend strength on the daily timeframe.
# Works in both bull and bear markets by following the dominant trend as defined by higher timeframe.
# Entry: Go long when Alligator is bullish (Lips > Teeth > Jaw) and ADX > 25.
# Entry: Go short when Alligator is bearish (Lips < Teeth < Jaw) and ADX > 25.
# Exit: When Alligator lines become intertwined (trend weakening) or ADX < 20.

name = "6h_Williams_Alligator_ADX_Trend_1d"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # === Williams Alligator on 6h (13, 8, 5 SMMA with offsets) ===
    # SMMA (Smoothed Moving Average) implementation
    def smma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        result = np.full_like(arr, np.nan, dtype=float)
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    # Jaw (13-period SMMA, offset 8 bars)
    jaw = smma(close, 13)
    jaw = np.roll(jaw, 8)  # shift right by 8
    jaw[:8] = np.nan
    
    # Teeth (8-period SMMA, offset 5 bars)
    teeth = smma(close, 8)
    teeth = np.roll(teeth, 5)  # shift right by 5
    teeth[:5] = np.nan
    
    # Lips (5-period SMMA, offset 3 bars)
    lips = smma(close, 5)
    lips = np.roll(lips, 3)  # shift right by 3
    lips[:3] = np.nan
    
    # === ADX on 1d (14-period) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values (using Wilder's smoothing, similar to EMA but different factor)
    def wilders_smoothing(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        result = np.full_like(arr, np.nan, dtype=float)
        # First value is simple average
        valid_start = ~np.isnan(arr)
        if not np.any(valid_start):
            return result
        first_valid_idx = np.where(valid_start)[0][0]
        if first_valid_idx + period >= len(arr):
            return result
        result[first_valid_idx + period - 1] = np.nanmean(arr[first_valid_idx:first_valid_idx + period])
        # Subsequent values
        for i in range(first_valid_idx + period, len(arr)):
            if not np.isnan(arr[i]):
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
            else:
                result[i] = result[i-1]
        return result
    
    # Smooth TR, DM+, DM-
    atr = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, 14)
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Alligator conditions
        alligator_bullish = lips[i] > teeth[i] and teeth[i] > jaw[i]
        alligator_bearish = lips[i] < teeth[i] and teeth[i] < jaw[i]
        alligator_awake = alligator_bullish or alligator_bearish  # not intertwined
        
        # ADX trend strength
        strong_trend = adx_aligned[i] > 25
        weak_trend = adx_aligned[i] < 20
        
        if position == 0:
            # LONG: Alligator bullish and strong trend
            if alligator_bullish and strong_trend:
                signals[i] = 0.25
                position = 1
            # SHORT: Alligator bearish and strong trend
            elif alligator_bearish and strong_trend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Alligator becomes intertwined or trend weakens
            if not alligator_awake or weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Alligator becomes intertwined or trend weakens
            if not alligator_awake or weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals