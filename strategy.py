#!/usr/bin/env python3
"""
12h Williams Alligator with Volume Spike and 1d Trend Filter v1
Hypothesis: The Williams Alligator (three smoothed moving averages) identifies 
trending markets on 12h timeframe. When combined with volume spikes (>2x average) 
and strong 1d trend (ADX > 20), it captures sustained moves while avoiding 
false signals in ranging markets. Works in both bull and bear by requiring 
trend alignment and volume confirmation. Target: 12-37 trades/year.
"""

name = "12h_williams_alligator_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filters - call ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 14-period ADX for 1d
    # True Range
    tr1_1d = high_1d[1:] - low_1d[1:]
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))])
    
    # Directional Movement
    dm_plus_1d = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                          np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus_1d = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                           np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus_1d = np.concatenate([[0], dm_plus_1d])
    dm_minus_1d = np.concatenate([[0], dm_minus_1d])
    
    # Smoothed values
    tr14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    dm_plus_14_1d = pd.Series(dm_plus_1d).rolling(window=14, min_periods=14).sum().values
    dm_minus_14_1d = pd.Series(dm_minus_1d).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    di_plus_1d = 100 * dm_plus_14_1d / tr14_1d
    di_minus_1d = 100 * dm_minus_14_1d / tr14_1d
    
    # DX and ADX
    dx_1d = 100 * np.abs(di_plus_1d - di_minus_1d) / (di_plus_1d + di_minus_1d)
    adx_1d = pd.Series(dx_1d).rolling(window=14, min_periods=14).mean().values
    
    # Williams Alligator on 12h (Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA)
    # SMMA = smoothed moving average (similar to Wilder's smoothing)
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: (prev * (period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)  # Jaw (blue line)
    teeth = smma(close, 8)  # Teeth (red line)
    lips = smma(close, 5)   # Lips (green line)
    
    # Volume spike detector: current volume > 2 x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_1d[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i])):
            signals[i] = 0.0
            continue
        
        # Get aligned 1d ADX for current 12h bar
        adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)[i]
        
        # Regime filter: only trade in strong trending markets on daily
        strong_trend_1d = adx_1d_aligned > 20
        
        # Alligator alignment: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        alligator_long = lips[i] > teeth[i] and teeth[i] > jaw[i]
        alligator_short = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        if position == 1:  # Long position
            # Exit: trend weakens OR alligator alignment breaks
            if not strong_trend_1d or not alligator_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: trend weakens OR alligator alignment breaks
            if not strong_trend_1d or not alligator_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only trade with volume spike, strong 1d trend, and alligator alignment
            if volume_spike[i] and strong_trend_1d and alligator_long:
                position = 1
                signals[i] = 0.25
            elif volume_spike[i] and strong_trend_1d and alligator_short:
                position = -1
                signals[i] = -0.25
    
    return signals