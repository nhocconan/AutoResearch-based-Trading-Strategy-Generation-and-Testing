#!/usr/bin/env python3
"""
6h_Adaptive_Range_Breakout_1dTrend
Hypothesis: In 6h timeframe, price often consolidates in ranges before breaking out with momentum.
Using 1d ADX to filter trending vs ranging markets, we enter breakouts from 6h Donchian channels
only when the daily trend is strong (ADX > 25). In ranging markets (ADX < 20), we fade at
Donchian boundaries with mean-reversion. This adapts to market regime, reducing false breakouts
in chop and capturing momentum in trends. Volume confirmation adds institutional validation.
Designed for 6h to target 50-150 total trades over 4 years (12-37/year).
"""

name = "6h_Adaptive_Range_Breakout_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14) on daily
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_minus = np.concatenate([[np.nan], dm_minus])
    # Smoothed values
    def smoothed_avg(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value: simple average
        result[period-1] = np.nanmean(arr[1:period])
        # Subsequent values: Wilder smoothing
        for i in range(period, len(arr)):
            if not np.isnan(arr[i]) and not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    atr = smoothed_avg(tr, 14)
    dm_plus_smooth = smoothed_avg(dm_plus, 14)
    dm_minus_smooth = smoothed_avg(dm_minus, 14)
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = smoothed_avg(dx, 14)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 6h data for Donchian channels and volume
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period) on 6h
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if i >= window - 1:
                result[i] = np.nanmax(arr[i-window+1:i+1])
        return result
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if i >= window - 1:
                result[i] = np.nanmin(arr[i-window+1:i+1])
        return result
    donch_high = rolling_max(high, 20)
    donch_low = rolling_min(low, 20)
    
    # Volume average (20-period) for spike detection
    vol_avg = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 19:
            vol_avg[i] = np.nanmean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Donchian (20) and volume avg (20)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(vol_avg[i]) or np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: ADX > 25 = trending, ADX < 20 = ranging
        trending = adx_aligned[i] > 25
        ranging = adx_aligned[i] < 20
        
        # Volume filter: current volume > 1.5x average
        volume_spike = volume[i] > vol_avg[i] * 1.5
        
        if position == 0:
            if trending:
                # In trending market: breakout continuation
                if high[i] > donch_high[i] and volume_spike:
                    signals[i] = 0.25
                    position = 1
                elif low[i] < donch_low[i] and volume_spike:
                    signals[i] = -0.25
                    position = -1
            elif ranging:
                # In ranging market: mean reversion at boundaries
                if low[i] <= donch_low[i] and volume_spike:
                    signals[i] = 0.25
                    position = 1
                elif high[i] >= donch_high[i] and volume_spike:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: opposite Donchian touch or ADX drops to ranging
            if low[i] <= donch_low[i] or (ranging and adx_aligned[i] < 25):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: opposite Donchian touch or ADX drops to ranging
            if high[i] >= donch_high[i] or (ranging and adx_aligned[i] < 25):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals