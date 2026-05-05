#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d volume spike + 1d ADX trend filter
# Long when price breaks above 20-period high AND 1d volume > 2.0x 20-period average AND 1d ADX > 25 (trending)
# Short when price breaks below 20-period low AND 1d volume > 2.0x 20-period average AND 1d ADX > 25 (trending)
# Exit when price crosses back to midpoint of Donchian channel OR 1d ADX < 20 (range)
# Uses discrete sizing (0.30) to limit fee drag. Target: 20-40 trades/year per symbol.
# Donchian provides clear structure, volume spike confirms institutional participation,
# ADX filter ensures we only trade in trending markets to avoid whipsaws in chop.
# Works in bull markets via longs in uptrends and bear markets via shorts in downtrends.

name = "4h_Donchian20_VolumeSpike_1dADX_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for ADX and volume calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for ADX calculation
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[np.nan], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[np.nan], close_1d[:-1]]))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.concatenate([[np.nan], high_1d[:-1]])) > 
                       (np.concatenate([[np.nan], low_1d[:-1]]) - low_1d), 
                       np.maximum(high_1d - np.concatenate([[np.nan], high_1d[:-1]]), 0), 0)
    dm_minus = np.where((np.concatenate([[np.nan], low_1d[:-1]]) - low_1d) > 
                        (high_1d - np.concatenate([[np.nan], high_1d[:-1]])), 
                        np.maximum(np.concatenate([[np.nan], low_1d[:-1]]) - low_1d, 0), 0)
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan)
        if len(values) >= period:
            # First value is simple average
            result[period-1] = np.nansum(values[:period])
            # Subsequent values: Wilder's smoothing
            for i in range(period, len(values)):
                result[i] = result[i-1] - (result[i-1] / period) + values[i]
        return result
    
    period = 14
    tr_period = wilders_smoothing(tr, period)
    dm_plus_period = wilders_smoothing(dm_plus, period)
    dm_minus_period = wilders_smoothing(dm_minus, period)
    
    # DI+ and DI-
    di_plus = np.where(tr_period != 0, (dm_plus_period / tr_period) * 100, 0)
    di_minus = np.where(tr_period != 0, (dm_minus_period / tr_period) * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 
                  np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = wilders_smoothing(dx, period)
    
    # Trend conditions: ADX > 25 (trending), ADX < 20 (range)
    adx_trending = adx > 25
    adx_ranging = adx < 20
    
    # 1d volume spike: volume > 2.0x 20-period average
    if len(df_1d) >= 20:
        vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
        volume_filter = df_1d['volume'].values > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(len(df_1d), dtype=bool)
    
    # Align 1d indicators to 4h timeframe
    adx_trending_aligned = align_htf_to_ltf(prices, df_1d, adx_trending.astype(float))
    adx_ranging_aligned = align_htf_to_ltf(prices, df_1d, adx_ranging.astype(float))
    volume_filter_aligned = align_htf_to_ltf(prices, df_1d, volume_filter.astype(float))
    
    # Calculate 4h Donchian channel (20-period)
    if len(high) >= 20:
        donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_mid = (donchian_high + donchian_low) / 2
    else:
        donchian_high = np.full(n, np.nan)
        donchian_low = np.full(n, np.nan)
        donchian_mid = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or 
            np.isnan(adx_trending_aligned[i]) or 
            np.isnan(adx_ranging_aligned[i]) or 
            np.isnan(volume_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high AND volume spike AND 1d trending (ADX > 25)
            if (close[i] > donchian_high[i] and 
                volume_filter_aligned[i] > 0.5 and 
                adx_trending_aligned[i] > 0.5):
                signals[i] = 0.30
                position = 1
            # Short conditions: price breaks below Donchian low AND volume spike AND 1d trending (ADX > 25)
            elif (close[i] < donchian_low[i] and 
                  volume_filter_aligned[i] > 0.5 and 
                  adx_trending_aligned[i] > 0.5):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price crosses back to Donchian mid OR 1d ranging (ADX < 20)
            if (close[i] < donchian_mid[i] or 
                adx_ranging_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price crosses back to Donchian mid OR 1d ranging (ADX < 20)
            if (close[i] > donchian_mid[i] or 
                adx_ranging_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals