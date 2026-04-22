# Solution
#!/usr/bin/env python3
"""
Hypothesis: Daily Donchian breakout with weekly ADX trend filter.
Long when price breaks above Donchian(20) high, weekly ADX > 25, and volume above average.
Short when price breaks below Donchian(20) low, weekly ADX > 25, and volume above average.
Exit when price crosses back below/above Donchian midpoint or weekly ADX drops below 20.
Donchian provides clear breakout levels; weekly ADX filters for trending conditions.
Designed for low trade frequency by requiring breakout + trend + volume confirmation.
Works in both bull and bear markets by following weekly trend direction.
"""

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
    
    # Load daily data for Donchian and volume average - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Load weekly data for ADX trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Daily Donchian(20) channels
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_20 + low_20) / 2.0
    
    # Weekly ADX(14) calculation
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[:-1])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed TR and DM (Wilder's smoothing)
    def wilder_smoothing(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(arr[1:period])
        # Subsequent values: smoothed = (prev * (period-1) + current) / period
        for i in range(period, len(arr)):
            if not np.isnan(result[i-1]) and not np.isnan(arr[i]):
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
            else:
                result[i] = np.nan
        return result
    
    atr_1w = wilder_smoothing(tr, 14)
    dm_plus_smooth = wilder_smoothing(dm_plus, 14)
    dm_minus_smooth = wilder_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_1w != 0, dm_plus_smooth / atr_1w * 100, 0)
    di_minus = np.where(atr_1w != 0, dm_minus_smooth / atr_1w * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx_1w = wilder_smoothing(dx, 14)
    
    # Daily volume average (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align weekly ADX to daily timeframe
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after enough data for Donchian
        # Skip if data not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(donchian_mid[i]) or
            np.isnan(adx_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high, ADX > 25, volume above average
            if (close[i] > high_20[i] and 
                adx_1w_aligned[i] > 25 and 
                volume[i] > vol_ma_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low, ADX > 25, volume above average
            elif (close[i] < low_20[i] and 
                  adx_1w_aligned[i] > 25 and 
                  volume[i] > vol_ma_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price falls below Donchian mid OR ADX drops below 20
                if (close[i] < donchian_mid[i] or 
                    adx_1w_aligned[i] < 20):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price rises above Donchian mid OR ADX drops below 20
                if (close[i] > donchian_mid[i] or 
                    adx_1w_aligned[i] < 20):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Donchian20_1wADX25_VolumeFilter"
timeframe = "1d"
leverage = 1.0