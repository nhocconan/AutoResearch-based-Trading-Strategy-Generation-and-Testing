# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + 12h ADX trend filter
# - Long when price breaks above Donchian(20) high + volume spike + 12h ADX > 25
# - Short when price breaks below Donchian(20) low + volume spike + 12h ADX > 25
# - Exit when price returns to Donchian midpoint or ADX < 20
# - Donchian provides clear breakout levels, volume confirms institutional interest,
#   ADX filters for trending markets to avoid whipsaws in ranging conditions
# - Designed for 4h timeframe with selective entries to stay within trade limits

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for ADX calculation
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ADX on 12h timeframe
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.nan], np.maximum.reduce([tr1, tr2, tr3])])
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    def smoothed_ma(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) >= period:
            # Initial value
            result[period-1] = np.nansum(arr[1:period])
            # Wilder smoothing
            for i in range(period, len(arr)):
                if not np.isnan(result[i-1]) and not np.isnan(arr[i]):
                    result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    atr = smoothed_ma(tr, 14)
    dm_plus_smooth = smoothed_ma(dm_plus, 14)
    dm_minus_smooth = smoothed_ma(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = smoothed_ma(dx, 14)
    
    # Align ADX to 4h timeframe
    adx_4h = align_htf_to_ltf(prices, df_12h, adx)
    
    # Calculate Donchian channels on 4h
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    volume_4h = prices['volume'].values
    
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Volume spike detection (volume > 1.5x 20-period average)
    vol_ma = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_4h > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if NaN in indicators
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(adx_4h[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Breakout conditions
        breakout_up = close_4h[i] > donchian_high[i]
        breakout_down = close_4h[i] < donchian_low[i]
        
        # Return to midpoint conditions
        return_to_mid = np.abs(close_4h[i] - donchian_mid[i]) < 0.1 * (donchian_high[i] - donchian_low[i])
        
        if position == 0:
            # Long entry: breakout up + volume spike + ADX > 25
            if breakout_up and volume_spike[i] and adx_4h[i] > 25:
                signals[i] = 0.25
                position = 1
            # Short entry: breakout down + volume spike + ADX > 25
            elif breakout_down and volume_spike[i] and adx_4h[i] > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: return to midpoint OR ADX < 20 (trend weakening)
            if return_to_mid or adx_4h[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: return to midpoint OR ADX < 20
            if return_to_mid or adx_4h[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_VolumeSpike_ADXFilter"
timeframe = "4h"
leverage = 1.0