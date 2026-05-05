#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d ADX trend filter and volume confirmation
# Long when price breaks above Donchian upper AND 1d ADX > 25 AND volume > 2.0x 20-period average
# Short when price breaks below Donchian lower AND 1d ADX > 25 AND volume > 2.0x 20-period average
# Exit when price reverts to Donchian midpoint OR 1d ADX < 20 (trend weakening)
# Uses Donchian channels for clear breakout structure + ADX for trend strength + volume for confirmation
# Designed for 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# Timeframe: 6h (primary), HTF: 1d

name = "6h_Donchian20_1dADX_Trend_VolumeSpike_2.0x"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough data for ADX
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian(20) on 6h timeframe
    if len(high) >= 20:
        donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_mid = (donchian_high + donchian_low) / 2.0
    else:
        donchian_high = np.full(n, np.nan)
        donchian_low = np.full(n, np.nan)
        donchian_mid = np.full(n, np.nan)
    
    # Calculate 1d ADX(14)
    if len(high_1d) >= 14:
        # True Range
        tr1 = high_1d[1:] - low_1d[1:]
        tr2 = np.abs(high_1d[1:] - close_1d[:-1])
        tr3 = np.abs(low_1d[1:] - close_1d[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # First value is NaN
        
        # Directional Movement
        dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                           np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
        dm_plus = np.concatenate([[np.nan], dm_plus])
        dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                            np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
        dm_minus = np.concatenate([[np.nan], dm_minus])
        
        # Smoothed TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
        def wilders_smoothing(data, period):
            result = np.full_like(data, np.nan)
            if len(data) < period:
                return result
            # First value is simple average
            result[period-1] = np.nanmean(data[1:period])
            # Subsequent values: Wilder's smoothing
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
            return result
        
        atr = wilders_smoothing(tr, 14)
        dm_plus_smooth = wilders_smoothing(dm_plus, 14)
        dm_minus_smooth = wilders_smoothing(dm_minus, 14)
        
        # DI+ and DI-
        di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
        di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
        
        # DX and ADX
        dx = np.where((di_plus + di_minus) != 0, 
                      100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
        adx = wilders_smoothing(dx, 14)
    else:
        adx = np.full(len(close_1d), np.nan)
    
    # Align HTF indicators to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation on 6h (threshold: 2.0x for optimal frequency)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (2.0 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high AND ADX > 25 AND volume spike
            if (close[i] > donchian_high[i] and 
                adx_aligned[i] > 25 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND ADX > 25 AND volume spike
            elif (close[i] < donchian_low[i] and 
                  adx_aligned[i] > 25 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price reverts to midpoint OR ADX < 20 (trend weakening)
            if close[i] < donchian_mid[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reverts to midpoint OR ADX < 20 (trend weakening)
            if close[i] > donchian_mid[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals