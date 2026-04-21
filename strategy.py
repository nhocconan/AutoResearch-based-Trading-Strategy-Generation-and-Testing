# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
12h Donchian Breakout with 1d ADX Trend Filter and Volume Confirmation
Hypothesis: Donchian channel breakouts capture momentum in trending markets.
Using 12h timeframe for primary signals reduces trade frequency to avoid fee drag.
1d ADX > 25 filters for trending conditions, avoiding choppy markets.
Volume confirmation ensures breakouts have institutional participation.
Designed for 15-35 trades/year to minimize fee impact while capturing strong moves.
Works in bull/bear via ADX trend filter - only trades in confirmed trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX(14) on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    def _smooth(val, period):
        smoothed = np.full_like(val, np.nan, dtype=float)
        if len(val) < period:
            return smoothed
        smoothed[period-1] = np.nansum(val[:period])
        for i in range(period, len(val)):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + val[i]
        return smoothed
    
    atr = _smooth(tr, 14)
    dm_plus_smooth = _smooth(dm_plus, 14)
    dm_minus_smooth = _smooth(dm_minus, 14)
    
    # DI and DX
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = _smooth(dx, 14)
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Load 12h data ONCE before loop for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Donchian Channel (20-period) on 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate rolling max/min
    def _rolling_max(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(len(arr)):
            if i < window - 1:
                result[i] = np.nan
            else:
                result[i] = np.max(arr[i-window+1:i+1])
        return result
    
    def _rolling_min(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(len(arr)):
            if i < window - 1:
                result[i] = np.nan
            else:
                result[i] = np.min(arr[i-window+1:i+1])
        return result
    
    donchian_high = _rolling_max(high_12h, 20)
    donchian_low = _rolling_min(low_12h, 20)
    
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # Volume confirmation: 12h volume / 20-period average volume
    vol_ma_20 = np.full_like(df_12h['volume'].values, np.nan, dtype=float)
    vol_series = df_12h['volume'].values
    for i in range(len(vol_series)):
        if i < 19:
            vol_ma_20[i] = np.nan
        else:
            vol_ma_20[i] = np.mean(vol_series[i-19:i+1])
    
    vol_ratio_12h = vol_series / vol_ma_20
    vol_ratio_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        adx_val = adx_aligned[i]
        upper_band = donchian_high_aligned[i]
        lower_band = donchian_low_aligned[i]
        vol_ratio = vol_ratio_aligned[i]
        
        # Trend filter: ADX > 25 indicates trending market
        is_trending = adx_val > 25
        # Volume filter: volume > 1.5x average
        volume_ok = vol_ratio > 1.5
        
        if position == 0:
            # Enter long: price breaks above upper Donchian band + trend + volume
            if (price_close > upper_band and is_trending and volume_ok):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower Donchian band + trend + volume
            elif (price_close < lower_band and is_trending and volume_ok):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price returns to middle of Donchian channel or trend weakness
            middle_band = (upper_band + lower_band) / 2
            trend_weak = adx_val < 20  # ADX < 20 indicates weakening trend
            
            if position == 1 and (price_close < middle_band or trend_weak):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (price_close > middle_band or trend_weak):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_DonchianBreakout_1dADX_Trend_Volume"
timeframe = "12h"
leverage = 1.0