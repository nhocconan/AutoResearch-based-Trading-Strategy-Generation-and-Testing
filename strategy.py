#!/usr/bin/env python3
"""
6H_ADX_Alligator_Signal_Filter
Hypothesis: Combine ADX trend strength (ADX>25) with Williams Alligator (Jaw/Teeth/Lips) alignment for trend direction.
Long when: ADX>25 + Lips>Teeth>Jaw (bullish alignment).
Short when: ADX>25 + Lips<Teeth<Jaw (bearish alignment).
Use 1D Alligator for higher timeframe trend filter to avoid counter-trend trades.
6h timeframe targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
Works in both bull and bear markets by filtering for strong trending conditions only.
"""
name = "6H_ADX_Alligator_Signal_Filter"
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
    
    # Get 1D data for Alligator (Jaw, Teeth, Lips)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams Alligator on 1D
    # Jaw: 13-period SMMA shifted 8 bars
    # Teeth: 8-period SMMA shifted 5 bars  
    # Lips: 5-period SMMA shifted 3 bars
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full(len(arr), np.nan)
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (prev * (period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw_1d = smma(close_1d, 13)
    teeth_1d = smma(close_1d, 8)
    lips_1d = smma(close_1d, 5)
    
    # Shift the lines (Jaw +8, Teeth +5, Lips +3)
    jaw_1d = np.roll(jaw_1d, 8)
    teeth_1d = np.roll(teeth_1d, 5)
    lips_1d = np.roll(lips_1d, 3)
    # Set shifted values to NaN
    jaw_1d[:8] = np.nan
    teeth_1d[:5] = np.nan
    lips_1d[:3] = np.nan
    
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # Calculate ADX on 6H
    # +DM, -DM, TR
    high_diff = np.diff(high, prepend=high[0])
    low_diff = np.diff(low, prepend=low[0])
    high_diff[0] = 0
    low_diff[0] = 0
    
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(arr, period):
        """Wilder's smoothing (same as RSI smoothing)"""
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full(len(arr), np.nan)
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Wilder smoothing: previous * (period-1)/period + current/period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    period = 14
    atr = wilders_smoothing(tr, period)
    plus_di = 100 * wilders_smoothing(plus_dm, period) / atr
    minus_di = 100 * wilders_smoothing(minus_dm, period) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, period)
    
    # Volume filter: current volume > 1.3 x 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0
    
    start_idx = max(50, 20)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(adx[i]) or np.isnan(jaw_1d_aligned[i]) or 
            np.isnan(teeth_1d_aligned[i]) or np.isnan(lips_1d_aligned[i]) or
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 24 bars between trades (4 days on 6h TF) to reduce frequency
            if bars_since_exit < 24:
                continue
                
            # Check Alligator alignment
            bullish_alignment = (lips_1d_aligned[i] > teeth_1d_aligned[i] > jaw_1d_aligned[i])
            bearish_alignment = (lips_1d_aligned[i] < teeth_1d_aligned[i] < jaw_1d_aligned[i])
            
            # Long: ADX>25 + bullish alignment + volume filter
            if (adx[i] > 25 and bullish_alignment and volume_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_exit = 0
            # Short: ADX>25 + bearish alignment + volume filter
            elif (adx[i] > 25 and bearish_alignment and volume_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            # Exit if ADX weakens (<20) - trend losing strength
            if adx[i] < 20:
                exit_signal = True
            # Exit if Alligator alignment changes (lines cross)
            elif position == 1 and not (lips_1d_aligned[i] > teeth_1d_aligned[i] > jaw_1d_aligned[i]):
                exit_signal = True
            elif position == -1 and not (lips_1d_aligned[i] < teeth_1d_aligned[i] < jaw_1d_aligned[i]):
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals