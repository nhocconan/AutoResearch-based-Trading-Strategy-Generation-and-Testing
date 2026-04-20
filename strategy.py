#!/usr/bin/env python3
"""
6h_1d_ADX_Alligator_Signal_v1
Concept: 6h ADX trend strength combined with 1d Alligator (SMAs) for direction.
- Long: ADX > 25 (trending) + price above Alligator teeth (green alignment)
- Short: ADX > 25 + price below Alligator teeth (red alignment)
- Exit: ADX < 20 (weak trend) or price crosses Alligator jaw
- Uses 1d Alligator for higher timeframe alignment to reduce whipsaw
- Position sizing: 0.25
- Target: 20-50 trades/year (80-200 total over 4 years)
- Works in bull/bear: ADX filters ranging markets, Alligator provides trend direction
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_ADX_Alligator_Signal_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # === 6h: ADX for trend strength ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values (Wilder smoothing = EMA with alpha=1/period)
    def WilderSmooth(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        # First value is simple average
        if len(data) >= period:
            result[period-1] = np.nanmean(data[:period])
            for i in range(period, len(data)):
                if not np.isnan(data[i]) and not np.isnan(result[i-1]):
                    result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
                else:
                    result[i] = result[i-1]
        return result
    
    period = 14
    atr = WilderSmooth(tr, period)
    dm_plus_smooth = WilderSmooth(dm_plus, period)
    dm_minus_smooth = WilderSmooth(dm_minus, period)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 
                  100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = WilderSmooth(dx, period)
    
    # === 1d: Alligator (SMAs) ===
    # Alligator: Jaw (13 SMA), Teeth (8 SMA), Lips (5 SMA)
    close_1d = df_1d['close'].values
    # Calculate SMAs
    def sma(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if i >= window - 1:
                result[i] = np.mean(arr[i-window+1:i+1])
        return result
    
    jaw = sma(close_1d, 13)   # Blue line
    teeth = sma(close_1d, 8)  # Red line
    lips = sma(close_1d, 5)   # Green line
    
    # Align Alligator lines to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 34)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Get values
        adx_val = adx[i]
        close_val = close[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(adx_val) or np.isnan(jaw_val) or np.isnan(teeth_val) or 
            np.isnan(lips_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Strong trend (ADX > 25) + bullish alignment (Lips > Teeth > Jaw)
            if adx_val > 25 and lips_val > teeth_val > jaw_val and close_val > lips_val:
                signals[i] = 0.25
                position = 1
            # Enter short: Strong trend (ADX > 25) + bearish alignment (Lips < Teeth < Jaw)
            elif adx_val > 25 and lips_val < teeth_val < jaw_val and close_val < lips_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Weak trend (ADX < 20) or price crosses below Jaw
            if adx_val < 20 or close_val < jaw_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Weak trend (ADX < 20) or price crosses above Jaw
            if adx_val < 20 or close_val > jaw_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals