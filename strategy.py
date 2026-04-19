#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w ADX filter and volume confirmation
# Uses 1w ADX > 25 for trending regime to avoid chop, reducing false signals
# Donchian breakout captures breakouts in trending markets
# Volume confirmation (>1.5x 20-period average) ensures momentum
# Target: 15-25 trades/year per symbol with disciplined entries
name = "12h_Donchian20_1wADX_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w ADX for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate ADX on 1w data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = 0  # First period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), 
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)), 
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+
    tr14 = np.full_like(tr, np.nan, dtype=float)
    dm_plus_14 = np.full_like(dm_plus, np.nan, dtype=float)
    dm_minus_14 = np.full_like(dm_minus, np.nan, dtype=float)
    
    if len(tr) >= 14:
        tr14[13] = np.sum(tr[:14])
        dm_plus_14[13] = np.sum(dm_plus[:14])
        dm_minus_14[13] = np.sum(dm_minus[:14])
        for i in range(14, len(tr)):
            tr14[i] = tr14[i-1] - (tr14[i-1] / 14) + tr[i]
            dm_plus_14[i] = dm_plus_14[i-1] - (dm_plus_14[i-1] / 14) + dm_plus[i]
            dm_minus_14[i] = dm_minus_14[i-1] - (dm_minus_14[i-1] / 14) + dm_minus[i]
    
    # DI+ and DI-
    di_plus = np.full_like(tr, np.nan, dtype=float)
    di_minus = np.full_like(tr, np.nan, dtype=float)
    valid = ~np.isnan(tr14) & (tr14 != 0)
    di_plus[valid] = 100 * dm_plus_14[valid] / tr14[valid]
    di_minus[valid] = 100 * dm_minus_14[valid] / tr14[valid]
    
    # DX and ADX
    dx = np.full_like(tr, np.nan, dtype=float)
    di_sum = di_plus + di_minus
    valid_dx = ~np.isnan(di_sum) & (di_sum != 0)
    dx[valid_dx] = 100 * np.abs(di_plus[valid_dx] - di_minus[valid_dx]) / di_sum[valid_dx]
    
    adx = np.full_like(tr, np.nan, dtype=float)
    if len(dx) >= 14:
        adx[27] = np.nanmean(dx[14:28])  # First ADX at index 27
        for i in range(28, len(dx)):
            if not np.isnan(dx[i]) and not np.isnan(adx[i-1]):
                adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Donchian(20) on 12h data
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate Donchian channels
    dc_high = np.full_like(high_12h, np.nan, dtype=float)
    dc_low = np.full_like(low_12h, np.nan, dtype=float)
    
    for i in range(19, len(high_12h)):
        dc_high[i] = np.max(high_12h[i-19:i+1])
        dc_low[i] = np.min(low_12h[i-19:i+1])
    
    dc_high_aligned = align_htf_to_ltf(prices, df_12h, dc_high)
    dc_low_aligned = align_htf_to_ltf(prices, df_12h, dc_low)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = np.full_like(volume, np.nan, dtype=float)
    for i in range(19, len(volume)):
        volume_ma[i] = np.mean(volume[i-19:i+1])
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(dc_high_aligned[i]) or np.isnan(dc_low_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Only trade in trending markets (ADX > 25)
        if adx_aligned[i] <= 25:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: close breaks above Donchian high + volume confirmation
            if close[i] > dc_high_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: close breaks below Donchian low + volume confirmation
            elif close[i] < dc_low_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if close breaks below Donchian low
            if close[i] < dc_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if close breaks above Donchian high
            if close[i] > dc_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals