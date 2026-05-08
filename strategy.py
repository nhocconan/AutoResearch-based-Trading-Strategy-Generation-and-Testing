#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and 1d ADX(14) trend filter.
# Long when price breaks above Donchian high(20) AND 1d volume > 1.5x 20-period average AND 1d ADX > 25.
# Short when price breaks below Donchian low(20) AND 1d volume > 1.5x 20-period average AND 1d ADX > 25.
# Exit when price crosses back below/above Donchian mid-point (mean of high/low over 20 periods).
# Uses Donchian breakouts for clear structure, volume for conviction, ADX for trend strength.
# Target: 60-120 total trades over 4 years (15-30/year) for low fee drift and robust performance.

name = "4h_Donchian20_1dVolume_ADX25"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1d data for volume and ADX filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Donchian channels (20-period) on 4h price
    high_max20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_max20 + low_min20) / 2
    
    # 1d volume filter: current volume > 1.5x 20-period average
    vol_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    volume_filter = vol_1d > (1.5 * vol_ma20_1d_aligned)
    
    # 1d ADX(14) for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    atr = pd.Series(tr).ewm(span=14, adjust=False).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, adjust=False).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, adjust=False).mean().values
    
    # DI and DX
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=14, adjust=False).mean().values
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Sufficient warmup for Donchian
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_max20[i]) or np.isnan(low_min20[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above Donchian high, volume filter, ADX > 25
            long_cond = (close[i] > high_max20[i]) and volume_filter[i] and (adx_aligned[i] > 25)
            # Short conditions: break below Donchian low, volume filter, ADX > 25
            short_cond = (close[i] < low_min20[i]) and volume_filter[i] and (adx_aligned[i] > 25)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below Donchian mid-point
            if close[i] < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above Donchian mid-point
            if close[i] > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals