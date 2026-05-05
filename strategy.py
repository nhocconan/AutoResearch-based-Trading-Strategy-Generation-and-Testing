#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike and 1w ADX trend filter
# Long when price breaks above Donchian upper AND volume > 2.0x 20-period average AND 1w ADX > 25 (trending)
# Short when price breaks below Donchian lower AND volume > 2.0x 20-period average AND 1w ADX > 25 (trending)
# Exit when price crosses back to Donchian midpoint OR 1w ADX < 20 (range)
# Uses discrete sizing (0.25) to limit fee drag. Target: 20-50 trades/year per symbol.
# Donchian provides price channel structure, volume spike confirms momentum,
# 1w ADX filters for trending markets to avoid chop whipsaws.
# Works in bull markets via longs in uptrends and bear markets via shorts in downtrends.

name = "4h_Donchian20_VolumeSpike_1wADX_Trend"
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
    
    # Get 1d data ONCE before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-period average volume on 1d data
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = vol_1d > (2.0 * vol_ma_20_1d)
    
    # Align 1d volume spike to 4h timeframe
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    # Get 1w data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate ADX(14) on 1w data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w - low_1w)
    tr2 = np.abs(high_1w - np.concatenate([[np.nan], close_1w[:-1]]))
    tr3 = np.abs(low_1w - np.concatenate([[np.nan], close_1w[:-1]]))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.concatenate([[np.nan], high_1w[:-1]])) > 
                       (np.concatenate([[np.nan], low_1w[:-1]]) - low_1w),
                       np.maximum(high_1w - np.concatenate([[np.nan], high_1w[:-1]]), 0), 0)
    dm_minus = np.where((np.concatenate([[np.nan], low_1w[:-1]]) - low_1w) > 
                        (high_1w - np.concatenate([[np.nan], high_1w[:-1]])),
                        np.maximum(np.concatenate([[np.nan], low_1w[:-1]]) - low_1w, 0), 0)
    
    # Smoothed TR, DM+
    tr_period = 14
    tr_sum = pd.Series(tr).rolling(window=tr_period, min_periods=tr_period).sum().values
    dm_plus_sum = pd.Series(dm_plus).rolling(window=tr_period, min_periods=tr_period).sum().values
    dm_minus_sum = pd.Series(dm_minus).rolling(window=tr_period, min_periods=tr_period).sum().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_sum / tr_sum
    di_minus = 100 * dm_minus_sum / tr_sum
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=tr_period, min_periods=tr_period).mean().values
    
    # Trend filters
    trending_1w = adx > 25
    ranging_1w = adx < 20
    
    # Align 1w trend to 4h timeframe
    trending_1w_aligned = align_htf_to_ltf(prices, df_1w, trending_1w.astype(float))
    ranging_1w_aligned = align_htf_to_ltf(prices, df_1w, ranging_1w.astype(float))
    
    # Calculate Donchian(20) on 4h data
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
            np.isnan(volume_spike_1d_aligned[i]) or 
            np.isnan(trending_1w_aligned[i]) or 
            np.isnan(ranging_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high AND volume spike AND 1w trending
            if (close[i] > donchian_high[i] and 
                volume_spike_1d_aligned[i] > 0.5 and 
                trending_1w_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low AND volume spike AND 1w trending
            elif (close[i] < donchian_low[i] and 
                  volume_spike_1d_aligned[i] > 0.5 and 
                  trending_1w_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back to Donchian mid OR 1w ranging
            if (close[i] < donchian_mid[i] or 
                ranging_1w_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back to Donchian mid OR 1w ranging
            if (close[i] > donchian_mid[i] or 
                ranging_1w_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals