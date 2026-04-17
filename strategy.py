#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with volume confirmation and weekly ADX trend filter.
Long when price breaks above 20-bar high AND 12h volume > 1.5x 20-bar avg AND weekly ADX > 25 (trending).
Short when price breaks below 20-bar low AND 12h volume > 1.5x 20-bar avg AND weekly ADX > 25.
Exit when price touches 12-bar midpoint OR opposite Donchian level.
Uses 1w for ADX trend regime and 12h for execution, volume, and Donchian channels.
Designed to capture strong trends with volume confirmation across bull and bear markets.
Target: 12-30 trades/year per symbol.
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
    
    # Get 1w data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w ADX (14-period)
    # True Range
    tr1 = np.maximum(high_1w - low_1w, 
                     np.absolute(high_1w - np.roll(close_1w, 1)),
                     np.absolute(low_1w - np.roll(close_1w, 1)))
    tr1[0] = high_1w[0] - low_1w[0]
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w),
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)),
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
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
    
    # Get 12h data for Donchian channels and volume
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h Donchian channels (20-period)
    donch_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # Calculate 12h volume MA (20-period)
    vol_ma_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to primary timeframe (assumed 12h)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    donch_high_aligned = align_htf_to_ltf(prices, df_12h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_12h, donch_low)
    donch_mid_aligned = align_htf_to_ltf(prices, df_12h, donch_mid)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    
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
        
        # Volume confirmation: current 12h volume > 1.5x 20-bar average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20_aligned[i]
        
        # Trend filter: ADX > 25 indicates strong trend
        trending = adx_aligned[i] > 25
        
        # Breakout conditions
        breakout_high = close[i] > donch_high_aligned[i]
        breakout_low = close[i] < donch_low_aligned[i]
        
        # Exit conditions: touch midpoint or opposite Donchian level
        touch_mid = abs(close[i] - donch_mid_aligned[i]) < 0.001 * close[i]  # within 0.1%
        touch_opposite = (position == 1 and close[i] < donch_low_aligned[i]) or \
                         (position == -1 and close[i] > donch_high_aligned[i])
        
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
            # Exit long: touch midpoint or break below Donchian low
            if (touch_mid or touch_opposite):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: touch midpoint or break above Donchian high
            if (touch_mid or touch_opposite):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Volume_ADX_Trend"
timeframe = "12h"
leverage = 1.0