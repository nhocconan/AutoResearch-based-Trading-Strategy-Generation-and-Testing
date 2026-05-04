#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume spike + ADX regime filter
# Enters long when price breaks above Donchian upper channel with volume confirmation and ADX > 25 (trending)
# Enters short when price breaks below Donchian lower channel with volume confirmation and ADX > 25
# Uses 1d ADX for regime filtering to avoid whipsaws in ranging markets
# Designed for 20-50 trades/year (~80-200 total over 4 years) to minimize fee drag
# Donchian provides clear structure, volume confirms breakout validity, ADX filters for trending regimes
# Works in bull markets via breakouts and in bear markets via breakdowns when ADX confirms trend

name = "4h_Donchian20_VolumeSpike_1dADX25_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough data for ADX calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX(14)
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
    
    # Smoothed values using Wilder's smoothing (alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    
    # Initialize arrays
    atr = np.full_like(tr, np.nan)
    dm_plus_smooth = np.full_like(dm_plus, np.nan)
    dm_minus_smooth = np.full_like(dm_minus, np.nan)
    
    # First values (simple average)
    if len(tr) >= period:
        atr[period-1] = np.nanmean(tr[1:period])  # Skip first NaN
        dm_plus_smooth[period-1] = np.nanmean(dm_plus[1:period])
        dm_minus_smooth[period-1] = np.nanmean(dm_minus[1:period])
    
    # Wilder's smoothing for rest
    for i in range(period, len(tr)):
        atr[i] = alpha * tr[i] + (1 - alpha) * atr[i-1]
        dm_plus_smooth[i] = alpha * dm_plus[i] + (1 - alpha) * dm_plus_smooth[i-1]
        dm_minus_smooth[i] = alpha * dm_minus[i] + (1 - alpha) * dm_minus_smooth[i-1]
    
    # Directional Indicators
    di_plus = np.full_like(atr, np.nan)
    di_minus = np.full_like(atr, np.nan)
    dx = np.full_like(atr, np.nan)
    
    valid = ~np.isnan(atr) & (atr != 0)
    di_plus[valid] = 100 * dm_plus_smooth[valid] / atr[valid]
    di_minus[valid] = 100 * dm_minus_smooth[valid] / atr[valid]
    dx[valid] = 100 * np.abs(di_plus[valid] - di_minus[valid]) / (di_plus[valid] + di_minus[valid])
    
    # ADX is smoothed DX
    adx = np.full_like(dx, np.nan)
    if len(dx) >= period:
        adx[2*period-2] = np.nanmean(dx[period-1:2*period-1])  # First ADX value
        for i in range(2*period-1, len(dx)):
            adx[i] = alpha * dx[i] + (1 - alpha) * adx[i-1]
    
    # Align 1d ADX to 4h timeframe (wait for completed 1d bar)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Donchian channels (20-period) on 4h data
    donchian_window = 20
    upper_channel = np.full_like(high, np.nan)
    lower_channel = np.full_like(low, np.nan)
    
    for i in range(donchian_window-1, len(high)):
        upper_channel[i] = np.max(high[i-donchian_window+1:i+1])
        lower_channel[i] = np.min(low[i-donchian_window+1:i+1])
    
    # Calculate volume spike filter (20-period volume MA)
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(19, len(volume)):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    volume_spike = volume > (vol_ma_20 * 2.0)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(upper_channel[i]) or 
            np.isnan(lower_channel[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper channel AND volume spike AND ADX > 25
            if (close[i] > upper_channel[i] and 
                volume_spike[i] and 
                adx_1d_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower channel AND volume spike AND ADX > 25
            elif (close[i] < lower_channel[i] and 
                  volume_spike[i] and 
                  adx_1d_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price re-enters Donchian channel OR ADX falls below 20 (trend weakening)
            if (close[i] >= lower_channel[i] and close[i] <= upper_channel[i]) or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price re-enters Donchian channel OR ADX falls below 20
            if (close[i] >= lower_channel[i] and close[i] <= upper_channel[i]) or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals