#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d volume spike and ADX trend filter.
Long when price breaks above 20-period 12h Donchian high AND 1d volume > 1.5x 20-bar average AND ADX(14) > 25.
Short when price breaks below 20-period 12h Donchian low AND 1d volume > 1.5x 20-bar average AND ADX(14) > 25.
Exit when price touches the opposite Donchian level (midpoint for re-entry prevention).
Uses 1d for volume confirmation and ADX regime filter, 12h for execution and Donchian channels.
Designed to capture strong trending moves with volume confirmation in both bull and bear markets.
Target: 15-35 trades/year per symbol (60-140 total over 4 years).
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
    
    # Get 12h data for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 12h Donchian channels (20-period)
    donch_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Get 1d data for volume confirmation and ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume MA for confirmation
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d ADX(14) for trend regime filter
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
    # Smoothed TR, DM+ , DM-
    tr14 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr14
    di_minus = 100 * dm_minus_14 / tr14
    # DX and ADX
    dx = 100 * np.absolute(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    # Handle division by zero
    adx = np.where((di_plus + di_minus) > 0, adx, 0.0)
    
    # Align all 1d and 12h indicators to 12h timeframe (primary)
    donch_high_aligned = align_htf_to_ltf(prices, df_12h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_12h, donch_low)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # need enough for indicators to warm up
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-bar average
        # Note: we use 1d volume at the 12h bar's timestamp (aligned)
        # Since we don't have 1d volume at 12h resolution, we use the latest available 1d volume
        # In practice, we check if the 1d volume (from the completed 1d bar) confirms
        volume_confirmed = volume_1d[i // 12] > 1.5 * vol_ma_20[i // 12] if i // 12 < len(volume_1d) else False
        
        # Regime filter: ADX > 25 indicates trending market
        trending_market = adx_aligned[i] > 25
        
        # Breakout conditions
        breakout_high = close[i] > donch_high_aligned[i]
        breakout_low = close[i] < donch_low_aligned[i]
        
        # Exit conditions: touch opposite Donchian level (prevent immediate re-entry)
        touch_opposite = (position == 1 and close[i] < donch_low_aligned[i]) or \
                         (position == -1 and close[i] > donch_high_aligned[i])
        
        if position == 0:
            # Long: break above Donchian high with volume confirmation and trending market
            if (breakout_high and volume_confirmed and trending_market):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with volume confirmation and trending market
            elif (breakout_low and volume_confirmed and trending_market):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: touch or break below Donchian low
            if touch_opposite:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: touch or break above Donchian high
            if touch_opposite:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Volume_ADX_Trend"
timeframe = "12h"
leverage = 1.0