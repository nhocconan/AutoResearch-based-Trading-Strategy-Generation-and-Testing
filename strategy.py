#!/usr/bin/env python3
# 6h_ADX_Trend_Strength_With_Volume_Filter
# Hypothesis: ADX > 25 indicates strong trend, +DI > -DI for long, -DI > +DI for short.
# Volume must be above 20-period average to confirm breakout strength.
# Works in bull markets by catching uptrends, in bear markets by catching downtrends.
# Uses 1-day ADX as higher timeframe filter to avoid counter-trend trades.
# Targets 15-35 trades per year on 6h timeframe with position size 0.25.

name = "6h_ADX_Trend_Strength_With_Volume_Filter"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 14-period ADX and DI on 1d
    # True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((df_1d['high'] - df_1d['high'].shift(1)) > (df_1d['low'].shift(1) - df_1d['low']),
                       np.maximum(df_1d['high'] - df_1d['high'].shift(1), 0), 0)
    dm_minus = np.where((df_1d['low'].shift(1) - df_1d['low']) > (df_1d['high'] - df_1d['high'].shift(1)),
                        np.maximum(df_1d['low'].shift(1) - df_1d['low'], 0), 0)
    
    # Smooth TR, DM+ and DM- with Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(arr[:period]) / period
        # Wilder smoothing: today = (yesterday * (period-1) + today) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    tr_smoothed = wilder_smooth(tr, 14)
    dm_plus_smoothed = wilder_smooth(dm_plus, 14)
    dm_minus_smoothed = wilder_smooth(dm_minus, 14)
    
    # Avoid division by zero
    tr_smoothed[tr_smoothed == 0] = 1e-10
    
    di_plus = 100 * dm_plus_smoothed / tr_smoothed
    di_minus = 100 * dm_minus_smoothed / tr_smoothed
    
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = wilder_smooth(dx, 14)
    
    # Get ADX and DI direction from 1d
    adx_1d = adx
    di_plus_1d = di_plus
    di_minus_1d = di_minus
    
    # Align 1d indicators to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    di_plus_1d_aligned = align_htf_to_ltf(prices, df_1d, di_plus_1d)
    di_minus_1d_aligned = align_htf_to_ltf(prices, df_1d, di_minus_1d)
    
    # Calculate 20-period volume average on 6h
    vol_ma = np.full_like(volume, np.nan, dtype=float)
    for i in range(20, len(volume)):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Warmup for 1d ADX (14+14+14=42 approx) and 20-period vol MA
    
    for i in range(start_idx, n):
        if np.isnan(adx_1d_aligned[i]) or np.isnan(di_plus_1d_aligned[i]) or np.isnan(di_minus_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend strength filter: ADX > 25 indicates strong trend
        strong_trend = adx_1d_aligned[i] > 25
        
        # Volume confirmation: current volume above average
        volume_filter = volume[i] > vol_ma[i]
        
        if position == 0:
            # Long entry: +DI > -DI, strong trend, and volume confirmation
            if di_plus_1d_aligned[i] > di_minus_1d_aligned[i] and strong_trend and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short entry: -DI > +DI, strong trend, and volume confirmation
            elif di_minus_1d_aligned[i] > di_plus_1d_aligned[i] and strong_trend and volume_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend weakens or reverses
            if di_plus_1d_aligned[i] <= di_minus_1d_aligned[i] or not strong_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend weakens or reverses
            if di_minus_1d_aligned[i] <= di_plus_1d_aligned[i] or not strong_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals