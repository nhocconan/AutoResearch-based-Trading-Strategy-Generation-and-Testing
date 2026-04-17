#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with volume confirmation and 1d ADX trend filter.
Long when price breaks above 20-bar high AND 4h volume > 1.5x 20-bar avg AND 1d ADX > 25 (trending).
Short when price breaks below 20-bar low AND 4h volume > 1.5x 20-bar avg AND 1d ADX > 25.
Exit when price touches 4h Donchian midpoint.
Uses 1d for ADX trend regime and 4h for execution, volume, and Donchian channels.
Designed to capture strong trends with volume confirmation across bull and bear markets.
Target: 19-50 trades/year per symbol.
"""

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
    volume = prices['volume'].values
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX (14-period)
    # True Range
    tr1 = np.maximum(high_1d - low_1d, 
                     np.absolute(high_1d - np.roll(close_1d, 1)),
                     np.absolute(low_1d - np.roll(close_1d, 1)))
    tr1[0] = high_1d[0] - low_1d[0]
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    # Smoothed TR, DM+-, DX
    tr14 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    dm_plus_sum = pd.Series(dm_plus_14).rolling(window=14, min_periods=14).sum().values
    dm_minus_sum = pd.Series(dm_minus_14).rolling(window=14, min_periods=14).sum().values
    tr14_sum = pd.Series(tr14).rolling(window=14, min_periods=14).sum().values
    dx = 100 * np.abs(dm_plus_sum - dm_minus_sum) / (dm_plus_sum + dm_minus_sum + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Get 4h data for Donchian channels and volume
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate 4h Donchian channels (20-period)
    donch_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # Calculate 4h volume MA (20-period)
    vol_ma_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to primary timeframe (4h)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    donch_high_aligned = align_htf_to_ltf(prices, df_4h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_4h, donch_low)
    donch_mid_aligned = align_htf_to_ltf(prices, df_4h, donch_mid)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup period
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(donch_high_aligned[i]) or
            np.isnan(donch_low_aligned[i]) or
            np.isnan(donch_mid_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x 20-bar average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20_aligned[i]
        
        # Trend filter: ADX > 25 indicates strong trend
        trending = adx_aligned[i] > 25
        
        # Breakout conditions
        breakout_high = close[i] > donch_high_aligned[i]
        breakout_low = close[i] < donch_low_aligned[i]
        
        # Exit conditions: touch midpoint
        touch_mid = abs(close[i] - donch_mid_aligned[i]) < 0.001 * close[i]  # within 0.1%
        
        if position == 0:
            # Long: break above Donchian high with volume confirmation and trend
            if (breakout_high and volume_confirmed and trending):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with volume confirmation and trend
            elif (breakout_low and volume_confirmed and trending):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: touch midpoint
            if touch_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: touch midpoint
            if touch_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Volume_1dADX_Trend"
timeframe = "4h"
leverage = 1.0