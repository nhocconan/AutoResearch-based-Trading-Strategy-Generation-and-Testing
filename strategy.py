#!/usr/bin/env python3
"""
6h_1d_adx_alligator_v1
Hypothesis: 6-hour strategy combining ADX trend strength with Williams Alligator for entry timing on 1d trend.
In bull markets: ADX>25 + price above Alligator teeth (green) = long.
In bear markets: ADX>25 + price below Alligator teeth (red) = short.
Uses 1d Alligator for trend filter, 6h ADX for entry timing to avoid whipsaws.
Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
Works in both bull/bear by requiring strong trend (ADX>25) and aligning with 1d Alligator direction.
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
    
    # Get 1d data for Alligator (trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Alligator on 1d: Jaw (13), Teeth (8), Lips (5) SMMA
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full(len(arr), np.nan)
        result = np.full(len(arr), np.nan)
        sma = np.nansum(arr[:period]) / period
        result[period-1] = sma
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw_1d = smma(close_1d, 13)
    teeth_1d = smma(close_1d, 8)
    lips_1d = smma(close_1d, 5)
    
    # Align Alligator lines to 6h
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # 6h ADX for trend strength
    def calculate_adx(high, low, close, period=14):
        """Calculate ADX with proper smoothing"""
        if len(high) < period + 1:
            return np.full(len(high), np.nan)
        
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr1[0] = tr2[0] = tr3[0] = np.nan
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        dm_plus[0] = dm_minus[0] = np.nan
        
        # Smoothed TR and DM
        def smooth_wilder(arr, period):
            """Wilder smoothing (same as EMA with alpha=1/period)"""
            if len(arr) < period:
                return np.full(len(arr), np.nan)
            result = np.full(len(arr), np.nan)
            # First value is simple average
            first_valid = ~np.isnan(arr)
            if not np.any(first_valid):
                return result
            first_idx = np.where(first_valid)[0][0]
            if first_idx + period >= len(arr):
                return result
            result[first_idx + period - 1] = np.nanmean(arr[first_idx:first_idx+period])
            for i in range(first_idx + period, len(arr)):
                if not np.isnan(arr[i]):
                    result[i] = (result[i-1] * (period-1) + arr[i]) / period
            return result
        
        tr_smooth = smooth_wilder(tr, period)
        dm_plus_smooth = smooth_wilder(dm_plus, period)
        dm_minus_smooth = smooth_wilder(dm_minus, period)
        
        # Directional Indicators
        plus_di = 100 * dm_plus_smooth / tr_smooth
        minus_di = 100 * dm_minus_smooth / tr_smooth
        
        # DX and ADX
        dx = np.where(tr_smooth != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
        adx = smooth_wilder(dx, period)
        
        return adx
    
    adx_6h = calculate_adx(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(jaw_1d_aligned[i]) or np.isnan(teeth_1d_aligned[i]) or 
            np.isnan(lips_1d_aligned[i]) or np.isnan(adx_6h[i])):
            signals[i] = 0.0
            continue
        
        # Determine Alligator alignment on 1d
        # Green (bullish): Lips > Teeth > Jaw
        # Red (bearish): Jaw > Teeth > Lips
        alligator_bullish = (lips_1d_aligned[i] > teeth_1d_aligned[i] > jaw_1d_aligned[i])
        alligator_bearish = (jaw_1d_aligned[i] > teeth_1d_aligned[i] > lips_1d_aligned[i])
        
        # Strong trend filter
        strong_trend = adx_6h[i] > 25
        
        # Long: strong trend + bullish Alligator
        if strong_trend and alligator_bullish and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: strong trend + bearish Alligator
        elif strong_trend and alligator_bearish and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: trend weakens or Alligator reverses
        elif position == 1 and (not strong_trend or not alligator_bullish):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (not strong_trend or not alligator_bearish):
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

name = "6h_1d_adx_alligator_v1"
timeframe = "6h"
leverage = 1.0