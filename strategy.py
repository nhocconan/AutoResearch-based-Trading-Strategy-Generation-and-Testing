#!/usr/bin/env python3
"""
6h Weekly Pivot R4/S4 Breakout with 1d ADX Trend Filter and Volume Spike.
Long when: 1) Price breaks above R4 (weekly resistance), 2) ADX > 25 (trending), 3) Volume > 2x 20-period average.
Short when: 1) Price breaks below S4 (weekly support), 2) ADX > 25 (trending), 3) Volume > 2x 20-period average.
Exit when price returns to weekly pivot point (mean reversion) or ADX < 20 (range).
Designed for 6h timeframe: targets 50-150 total trades over 4 years (12-37/year).
"""

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
    volume = prices['volume'].values
    
    # Get weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get daily data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Weekly high, low, close for pivot calculation
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot and R4/S4 levels
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r4_1w = close_1w + (high_1w - low_1w) * 1.1 / 2  # R4 = Close + 1.1*(H-L)/2
    s4_1w = close_1w - (high_1w - low_1w) * 1.1 / 2  # S4 = Close - 1.1*(H-L)/2
    
    # Align weekly levels to 6h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    # Daily ADX calculation (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align length
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    def _wilder_smoothing(arr, period):
        """Wilder's smoothing (same as RSI)"""
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(arr[1:period])  # skip index 0 (nan)
        # Rest is Wilder smoothing
        for i in range(period, len(arr)):
            if not np.isnan(result[i-1]) and not np.isnan(arr[i]):
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
            else:
                result[i] = np.nan
        return result
    
    atr = _wilder_smoothing(tr, 14)
    dm_plus_smooth = _wilder_smoothing(dm_plus, 14)
    dm_minus_smooth = _wilder_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 
                  100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = _wilder_smoothing(dx, 14)
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume filter: volume > 2x 20-period average
    vol_ma_20 = np.full(n, np.nan, dtype=np.float64)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need weekly pivot (need 1 week), daily ADX (14+14=28), volume MA (20)
    start_idx = max(28, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(pivot_1w_aligned[i]) or np.isnan(r4_1w_aligned[i]) or 
            np.isnan(s4_1w_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current values
        price = close[i]
        adx_val = adx_aligned[i]
        pivot_level = pivot_1w_aligned[i]
        r4_level = r4_1w_aligned[i]
        s4_level = s4_1w_aligned[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Trend and volume filters
        trending = adx_val > 25
        vol_filter = vol_now > 2.0 * vol_avg
        
        if position == 0:
            # Long: price breaks above R4 + trending + volume spike
            if price > r4_level and trending and vol_filter:
                signals[i] = size
                position = 1
            # Short: price breaks below S4 + trending + volume spike
            elif price < s4_level and trending and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to pivot (mean reversion) or trend weakens (ADX < 20)
            if price <= pivot_level or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to pivot (mean reversion) or trend weakens (ADX < 20)
            if price >= pivot_level or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Weekly_Pivot_R4S4_Breakout_1dADX_Trend_Volume"
timeframe = "6h"
leverage = 1.0