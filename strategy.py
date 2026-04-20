#!/usr/bin/env python3
# 4h_PriceChannel_VolumeTrend_Strategy
# Hypothesis: Price channel breakouts (Donchian) combined with trend (ADX) and volume confirmation
# capture momentum in both bull and bear markets. Volume ensures institutional participation.
# ADX > 25 filters for trending markets. Position size 0.25 to manage risk.
# Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag.

name = "4h_PriceChannel_VolumeTrend_Strategy"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for calculations (same timeframe)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    donchian_high = np.full_like(high_4h, np.nan)
    donchian_low = np.full_like(high_4h, np.nan)
    
    for i in range(len(high_4h)):
        if i >= 19:  # 20-period lookback
            donchian_high[i] = np.max(high_4h[i-19:i+1])
            donchian_low[i] = np.min(low_4h[i-19:i+1])
    
    # Align Donchian levels to LTF
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Calculate ADX (14-period) for trend strength
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth TR and DM
    tr_sum = np.full_like(high, np.nan)
    dm_plus_sum = np.full_like(high, np.nan)
    dm_minus_sum = np.full_like(high, np.nan)
    
    for i in range(len(high)):
        if i >= 13:  # 14-period smoothing
            tr_sum[i] = np.nansum(tr[i-13:i+1])
            dm_plus_sum[i] = np.nansum(dm_plus[i-13:i+1])
            dm_minus_sum[i] = np.nansum(dm_minus[i-13:i+1])
    
    # Directional Indicators
    di_plus = np.full_like(high, np.nan)
    di_minus = np.full_like(high, np.nan)
    dx = np.full_like(high, np.nan)
    
    valid = ~np.isnan(tr_sum) & (tr_sum != 0)
    di_plus[valid] = 100 * dm_plus_sum[valid] / tr_sum[valid]
    di_minus[valid] = 100 * dm_minus_sum[valid] / tr_sum[valid]
    dx[valid] = 100 * np.abs(di_plus[valid] - di_minus[valid]) / (di_plus[valid] + di_minus[valid])
    
    # ADX (smoothed DX)
    adx = np.full_like(high, np.nan)
    for i in range(len(high)):
        if i >= 27:  # 14 + 13 for ADX smoothing
            valid_dx = dx[i-13:i+1]
            if not np.all(np.isnan(valid_dx)):
                adx[i] = np.nanmean(valid_dx)
    
    # Volume confirmation: volume > 1.2x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma20 * 1.2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(28, 20)  # Ensure ADX and Donchian are calculated
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(adx[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high + ADX > 25 + volume confirmation
            if close[i] > donchian_high_aligned[i] and adx[i] > 25 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + ADX > 25 + volume confirmation
            elif close[i] < donchian_low_aligned[i] and adx[i] > 25 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below Donchian low or ADX weakens
            if close[i] < donchian_low_aligned[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above Donchian high or ADX weakens
            if close[i] > donchian_high_aligned[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals