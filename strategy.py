#!/usr/bin/env python3
# Hypothesis: 12h Donchian(20) breakout with 1d volume spike and ADX regime filter.
# Long when price breaks above 20-period 12h Donchian high AND 1d volume > 2.0x 20-period average AND 1d ADX > 25 (trending market)
# Short when price breaks below 20-period 12h Donchian low AND 1d volume > 2.0x 20-period average AND 1d ADX > 25
# Exit when price retraces to the midpoint of the 12h Donchian channel OR ADX < 20 (range market)
# Uses 12h for primary timeframe to limit trades, 1d for volume/ADX filters to avoid noise.
# Target: 50-150 total trades over 4 years (12-37/year). Works in bull via breakout continuation, bear via faded rallies.

name = "12h_Donchian20_1dVolume_ADX_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 12h data for Donchian channel calculation
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 20-period Donchian channel on 12h
    donchian_high_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_mid_12h = (donchian_high_12h + donchian_low_12h) / 2.0
    
    # Get 1d data for volume and ADX filters
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 20-period average volume on 1d
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (2.0 * vol_ma_1d)
    
    # Calculate 14-period ADX on 1d
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+ and DM- using Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(values, period):
        result = np.zeros_like(values)
        result[period-1] = np.nansum(values[:period])  # First value is simple average
        for i in range(period, len(values)):
            result[i] = result[i-1] - (result[i-1] / period) + values[i]
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    dm_plus_smoothed = wilders_smoothing(dm_plus, 14)
    dm_minus_smoothed = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smoothed / atr_1d
    di_minus = 100 * dm_minus_smoothed / atr_1d
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where((di_plus + di_minus) == 0, 0, dx)  # Avoid division by zero
    adx_1d = wilders_smoothing(dx, 14)
    
    # Align HTF indicators to 12h timeframe
    donchian_high_12h_aligned = align_htf_to_ltf(prices, df_12h, donchian_high_12h)
    donchian_low_12h_aligned = align_htf_to_ltf(prices, df_12h, donchian_low_12h)
    donchian_mid_12h_aligned = align_htf_to_ltf(prices, df_12h, donchian_mid_12h)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_12h_aligned[i]) or np.isnan(donchian_low_12h_aligned[i]) or
            np.isnan(volume_spike_1d_aligned[i]) or np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian high AND volume spike AND ADX > 25 (trending)
            if close[i] > donchian_high_12h_aligned[i] and volume_spike_1d_aligned[i] and adx_1d_aligned[i] > 25:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low AND volume spike AND ADX > 25 (trending)
            elif close[i] < donchian_low_12h_aligned[i] and volume_spike_1d_aligned[i] and adx_1d_aligned[i] > 25:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price retraces to midpoint OR ADX < 20 (range market)
            if close[i] < donchian_mid_12h_aligned[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price retraces to midpoint OR ADX < 20 (range market)
            if close[i] > donchian_mid_12h_aligned[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals