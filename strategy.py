#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 12h ADX trend filter
# Long when price breaks above 4h Donchian upper band with volume >1.5x 20-period average and 12h ADX > 25 (trending market)
# Short when price breaks below 4h Donchian lower band with volume >1.5x 20-period average and 12h ADX > 25
# Exit when price crosses the 4h Donchian midline
# ADX filter ensures we only trade in trending conditions, avoiding choppy markets
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h and 12h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 4h Donchian channel (20-period lookback)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Calculate 12h ADX (14-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with original index
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth TR, DM+, DM- using Wilder's smoothing (alpha = 1/period)
    atr = np.full_like(tr, np.nan, dtype=float)
    dm_plus_smooth = np.full_like(dm_plus, np.nan, dtype=float)
    dm_minus_smooth = np.full_like(dm_minus, np.nan, dtype=float)
    
    # Initial values (simple average of first 14 periods)
    if len(tr) >= 14:
        atr[13] = np.nanmean(tr[1:15])
        dm_plus_smooth[13] = np.nanmean(dm_plus[1:15])
        dm_minus_smooth[13] = np.nanmean(dm_minus[1:15])
        
        # Wilder smoothing for rest
        for i in range(14, len(tr)):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * 13 + dm_plus[i]) / 14
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * 13 + dm_minus[i]) / 14
    
    # Calculate DI+ and DI-
    di_plus = np.full_like(atr, np.nan, dtype=float)
    di_minus = np.full_like(atr, np.nan, dtype=float)
    dx = np.full_like(atr, np.nan, dtype=float)
    
    for i in range(14, len(atr)):
        if atr[i] != 0:
            di_plus[i] = (dm_plus_smooth[i] / atr[i]) * 100
            di_minus[i] = (dm_minus_smooth[i] / atr[i]) * 100
            if (di_plus[i] + di_minus[i]) != 0:
                dx[i] = (np.abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])) * 100
    
    # Calculate ADX (smoothed DX)
    adx = np.full_like(dx, np.nan, dtype=float)
    if len(dx) >= 27:  # Need 14 for DX + 14 more for ADX smoothing
        adx[26] = np.nanmean(dx[14:28])  # First ADX value
        for i in range(27, len(dx)):
            if not np.isnan(dx[i]):
                adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Calculate 4h volume average (20-period)
    vol_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 4h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_4h, donchian_middle)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50  # for 20-period Donchian and ADX calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_4h_current = volume[i]  # Current 4h volume
        
        if position == 0:
            # Long setup: break above Donchian upper with volume confirmation and ADX > 25
            if (price > donchian_upper_aligned[i] and 
                vol_4h_current > 1.5 * vol_ma_4h_aligned[i] and  # Volume confirmation
                adx_aligned[i] > 25):                          # Trending market filter
                position = 1
                signals[i] = position_size
            # Short setup: break below Donchian lower with volume confirmation and ADX > 25
            elif (price < donchian_lower_aligned[i] and 
                  vol_4h_current > 1.5 * vol_ma_4h_aligned[i] and  # Volume confirmation
                  adx_aligned[i] > 25):                          # Trending market filter
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian middle
            if price < donchian_middle_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above Donchian middle
            if price > donchian_middle_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Donchian_ADX_Volume"
timeframe = "4h"
leverage = 1.0