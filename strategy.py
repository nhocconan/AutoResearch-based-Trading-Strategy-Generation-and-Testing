#!/usr/bin/env python3
"""
6h_1d_ADX_Alligator_Triangle
Hypothesis: Use 1d ADX trend filter with Williams Alligator on 6h for entry timing.
- ADX > 25 on 1d indicates trending market (trend follow)
- Williams Alligator (Jaw=13, Teeth=8, Lips=5) on 6h: 
  Long when Lips > Teeth > Jaw (bullish alignment)
  Short when Lips < Teeth < Jaw (bearish alignment)
- Exit when Alligator lines re-cross (loss of alignment)
- Works in both bull/bear by only taking trades in trending regimes (ADX filter)
- Targets 20-40 trades/year with disciplined entries
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for ADX (trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan)
        if len(values) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(values[1:period]) 
        # Subsequent values: smoothed = prev - (prev/period) + current
        for i in range(period, len(values)):
            if not np.isnan(result[i-1]) and not np.isnan(values[i]):
                result[i] = result[i-1] - (result[i-1]/period) + values[i]
        return result
    
    period = 14
    atr = wilders_smoothing(tr, period)
    dm_plus_smooth = wilders_smoothing(dm_plus, period)
    dm_minus_smooth = wilders_smoothing(dm_minus, period)
    
    # DI+ and DI-
    di_plus = np.where(atr > 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr > 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, period)
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Williams Alligator on 6h (Jaw=13, Teeth=8, Lips=5)
    close = prices['close'].values
    
    def sma(values, window):
        result = np.full_like(values, np.nan)
        for i in range(window-1, len(values)):
            result[i] = np.mean(values[i-window+1:i+1])
        return result
    
    jaw = sma(close, 13)  # Jaw (Blue)
    teeth = sma(close, 8)  # Teeth (Red)
    lips = sma(close, 5)   # Lips (Green)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: ADX > 25 indicates trending market
        is_trending = adx_aligned[i] > 25
        
        if position == 0 and is_trending:
            # Long: Lips > Teeth > Jaw (bullish alignment)
            if lips[i] > teeth[i] and teeth[i] > jaw[i]:
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (bearish alignment)
            elif lips[i] < teeth[i] and teeth[i] < jaw[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: loss of bullish alignment
            if not (lips[i] > teeth[i] and teeth[i] > jaw[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: loss of bearish alignment
            if not (lips[i] < teeth[i] and teeth[i] < jaw[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1d_ADX_Alligator_Triangle"
timeframe = "6h"
leverage = 1.0