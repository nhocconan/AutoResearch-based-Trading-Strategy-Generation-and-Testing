#!/usr/bin/env python3
"""
Hypothesis: 4-hour Donchian channel breakout with 1-day volume confirmation and ADX trend filter.
Long when price breaks above 4h Donchian upper band (20-period) and 1-day volume > 50-day average volume and 1-day ADX > 25.
Short when price breaks below 4h Donchian lower band (20-period) and 1-day volume > 50-day average volume and 1-day ADX > 25.
Exit when price re-enters the Donchian channel (middle) or volume drops below average.
Donchian provides clear breakout levels; volume filter ensures institutional participation; ADX avoids ranging markets.
Works in both bull and bear markets by capturing strong trends with volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1-day data for volume and ADX filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1-day volume filter: 50-day average volume
    volume_1d = df_1d['volume'].values
    avg_vol_1d = pd.Series(volume_1d).rolling(window=50, min_periods=50).mean().values
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    
    # 1-day ADX calculation (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(data[:period]) / period
        # Subsequent values: smoothed = (prev_smoothed * (period-1) + current) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_1d = wilders_smooth(tr, 14)
    di_plus_1d = wilders_smooth(dm_plus, 14)
    di_minus_1d = wilders_smooth(dm_minus, 14)
    
    # Avoid division by zero
    dx_1d = np.where((di_plus_1d + di_minus_1d) != 0, 
                     100 * np.abs(di_plus_1d - di_minus_1d) / (di_plus_1d + di_minus_1d), 0)
    adx_1d = wilders_smooth(dx_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 4-hour Donchian channel (20-period)
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if i < window - 1:
                result[i] = np.nan
            else:
                result[i] = np.max(arr[i-window+1:i+1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if i < window - 1:
                result[i] = np.nan
            else:
                result[i] = np.min(arr[i-window+1:i+1])
        return result
    
    donchian_high = rolling_max(high, 20)
    donchian_low = rolling_min(low, 20)
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(avg_vol_1d_aligned[i]) or np.isnan(adx_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian upper band + volume filter + ADX trend
            if (close[i] > donchian_high[i] and 
                volume_1d[i] > avg_vol_1d_aligned[i] and 
                adx_1d_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower band + volume filter + ADX trend
            elif (close[i] < donchian_low[i] and 
                  volume_1d[i] > avg_vol_1d_aligned[i] and 
                  adx_1d_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price re-enters Donchian channel (below midpoint)
                if close[i] < donchian_mid[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price re-enters Donchian channel (above midpoint)
                if close[i] > donchian_mid[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian_Breakout_1dVolume_ADX_Filter"
timeframe = "4h"
leverage = 1.0