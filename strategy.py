#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d weekly Donchian breakout with weekly volume confirmation and weekly ADX trend filter.
# Long when price breaks above weekly Donchian upper channel (20) AND weekly volume > 1.3x 20-week average AND weekly ADX > 25.
# Short when price breaks below weekly Donchian lower channel (20) AND weekly volume > 1.3x 20-week average AND weekly ADX > 25.
# Exit when price crosses back inside the weekly Donchian channel.
# Uses weekly trend for direction, daily for execution to reduce whipsaw and capture major moves.
# Target: 30-100 total trades over 4 years (7-25/year) with low frequency to avoid fee drag.

name = "1d_WeeklyDonchian_Volume_ADX"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for Donchian channels, volume, and ADX
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 30:
        return np.zeros(n)
    
    # Weekly Donchian channels (20-period)
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    donchian_high = pd.Series(high_w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_w, donchian_low)
    
    # Weekly volume filter: current volume > 1.3x 20-week average
    vol_w = df_w['volume'].values
    vol_ma20_w = pd.Series(vol_w).rolling(window=20, min_periods=20).mean().values
    volume_filter_w = vol_w > (1.3 * vol_ma20_w)
    volume_filter_aligned = align_htf_to_ltf(prices, df_w, volume_filter_w)
    
    # Weekly ADX trend filter (14-period)
    # Calculate True Range
    high_w_arr = df_w['high'].values
    low_w_arr = df_w['low'].values
    close_w_arr = df_w['close'].values
    
    tr1 = high_w_arr - low_w_arr
    tr2 = np.abs(high_w_arr - np.roll(close_w_arr, 1))
    tr3 = np.abs(low_w_arr - np.roll(close_w_arr, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high_w_arr[0] - low_w_arr[0]  # First TR
    
    # Directional Movement
    dm_plus = np.where((high_w_arr - np.roll(high_w_arr, 1)) > (np.roll(low_w_arr, 1) - low_w_arr), 
                       np.maximum(high_w_arr - np.roll(high_w_arr, 1), 0), 0)
    dm_minus = np.where((np.roll(low_w_arr, 1) - low_w_arr) > (high_w_arr - np.roll(high_w_arr, 1)), 
                        np.maximum(np.roll(low_w_arr, 1) - low_w_arr, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # DI+ and DI-
    di_plus = np.where(tr14 != 0, 100 * dm_plus14 / tr14, 0)
    di_minus = np.where(tr14 != 0, 100 * dm_minus14 / tr14, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align weekly ADX to daily timeframe
    adx_aligned = align_htf_to_ltf(prices, df_w, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(volume_filter_aligned[i]) or np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above weekly Donchian high, volume filter, ADX > 25
            long_cond = (close[i] > donchian_high_aligned[i]) and volume_filter_aligned[i] and (adx_aligned[i] > 25)
            # Short conditions: price breaks below weekly Donchian low, volume filter, ADX > 25
            short_cond = (close[i] < donchian_low_aligned[i]) and volume_filter_aligned[i] and (adx_aligned[i] > 25)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below weekly Donchian low
            if close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above weekly Donchian high
            if close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals