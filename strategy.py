# Python 3.10
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d volume confirmation + 1d ADX trend filter
# In trending markets, price breaks Donchian bands with volume expansion.
# In ranging markets, ADX < 20 filters out false breakouts.
# Targets 15-30 trades per year (~60-120 total over 4 years) to minimize fee drag.

name = "12h_Donchian20_1dVolume_ADX"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels on 12h
    lookback = 20
    dc_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    dc_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Get 1d data for volume and ADX filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = df_1d['volume'].values / vol_ma
    vol_ratio = np.concatenate([[np.nan], vol_ratio[1:]])  # align length
    vol_confirm = align_htf_to_ltf(prices, df_1d, vol_ratio) > 1.5
    
    # ADX trend filter: ADX > 20 indicates trending market
    # Calculate ADX using Welles Wilder's method
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[1:period])
        # Subsequent values: smoothed = prev * (1 - 1/period) + current * (1/period)
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = result[i-1] * (1 - 1/period) + data[i] * (1/period)
            else:
                result[i] = np.nan
        return result
    
    atr_period = 14
    tr_smoothed = wilders_smoothing(tr, atr_period)
    dm_plus_smoothed = wilders_smoothing(dm_plus, atr_period)
    dm_minus_smoothed = wilders_smoothing(dm_minus, atr_period)
    
    # DI+ and DI-
    di_plus = np.where(tr_smoothed != 0, 100 * dm_plus_smoothed / tr_smoothed, 0)
    di_minus = np.where(tr_smoothed != 0, 100 * dm_minus_smoothed / tr_smoothed, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, atr_period)
    
    adx_confirm = align_htf_to_ltf(prices, df_1d, adx) > 20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, 30)  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(dc_high[i]) or np.isnan(dc_low[i]) or 
            np.isnan(vol_confirm[i]) or np.isnan(adx_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Donchian high with volume and trend confirmation
            if close[i] > dc_high[i] and vol_confirm[i] and adx_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low with volume and trend confirmation
            elif close[i] < dc_low[i] and vol_confirm[i] and adx_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian low or loses volume/trend confirmation
            if close[i] < dc_low[i] or not vol_confirm[i] or not adx_confirm[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian high or loses volume/trend confirmation
            if close[i] > dc_high[i] or not vol_confirm[i] or not adx_confirm[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals