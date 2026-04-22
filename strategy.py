#!/usr/bin/env python3
"""
Hypothesis: 12-hour Donchian(20) breakout with 1-day ADX trend filter and 1-day volume spike confirmation.
Long when price breaks above 20-period high and ADX > 25 and volume > 1.5x average.
Short when price breaks below 20-period low and ADX > 25 and volume > 1.5x average.
Exit when price returns to the midpoint of the Donchian channel.
Uses price channel breakouts for trend following, ADX to filter ranging markets,
and volume to confirm institutional participation. Works in trending markets (both bull and bear).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1-day data for ADX and volume filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX on 1-day data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth TR, DM+, DM- with Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        for i in range(len(data)):
            if np.isnan(data[i]):
                continue
            if np.isnan(result[i-1]) if i > 0 else True:
                result[i] = data[i]
            else:
                result[i] = (1 - alpha) * result[i-1] + alpha * data[i]
        return result
    
    period = 14
    atr_1d = wilder_smooth(tr, period)
    dm_plus_smooth = wilder_smooth(dm_plus, period)
    dm_minus_smooth = wilder_smooth(dm_minus, period)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr_1d
    di_minus = 100 * dm_minus_smooth / atr_1d
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx_1d = wilder_smooth(dx, period)
    
    # Average volume on 1-day
    volume_1d = df_1d['volume'].values
    avg_vol_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align ADX and volume average to 12h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    
    # Calculate Donchian channels on 12h data
    def rolling_max(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).max().values
    
    def rolling_min(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).min().values
    
    donch_high = rolling_max(high, 20)
    donch_low = rolling_min(low, 20)
    donch_mid = (donch_high + donch_low) / 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(avg_vol_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high, ADX > 25, volume > 1.5x average
            if (close[i] > donch_high[i] and 
                adx_1d_aligned[i] > 25 and 
                volume[i] > 1.5 * avg_vol_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low, ADX > 25, volume > 1.5x average
            elif (close[i] < donch_low[i] and 
                  adx_1d_aligned[i] > 25 and 
                  volume[i] > 1.5 * avg_vol_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit when price returns to the midpoint of the Donchian channel
            exit_signal = False
            
            if position == 1:
                # Exit long: Price falls below midpoint
                if close[i] < donch_mid[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price rises above midpoint
                if close[i] > donch_mid[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian_Breakout_1dADX_Volume_Filter"
timeframe = "12h"
leverage = 1.0