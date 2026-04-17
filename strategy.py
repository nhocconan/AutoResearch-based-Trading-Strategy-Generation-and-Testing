#!/usr/bin/env python3
"""
4h Donchian Breakout with Volume Spike and 1d ADX Trend Filter
Long: Price breaks above Donchian upper (20) + volume > 2x 4h volume SMA(20) + 1d ADX > 25
Short: Price breaks below Donchian lower (20) + volume > 2x 4h volume SMA(20) + 1d ADX > 25
Exit: Price retests Donchian midpoint or opposite stop via ATR
Uses Donchian channels for structure, volume for conviction, ADX for trend strength
Target: 20-30 trades/year per symbol (80-120 total over 4 years)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d ADX(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    tr14 = np.zeros_like(tr)
    dm_plus_14 = np.zeros_like(dm_plus)
    dm_minus_14 = np.zeros_like(dm_minus)
    
    # Initial values
    tr14[13] = np.nansum(tr[1:14])
    dm_plus_14[13] = np.nansum(dm_plus[1:14])
    dm_minus_14[13] = np.nansum(dm_minus[1:14])
    
    # Wilder's smoothing
    for i in range(14, len(tr)):
        tr14[i] = tr14[i-1] - (tr14[i-1] / 14) + tr[i]
        dm_plus_14[i] = dm_plus_14[i-1] - (dm_plus_14[i-1] / 14) + dm_plus[i]
        dm_minus_14[i] = dm_minus_14[i-1] - (dm_minus_14[i-1] / 14) + dm_minus[i]
    
    # DI and ADX
    di_plus = np.where(tr14 != 0, 100 * dm_plus_14 / tr14, 0)
    di_minus = np.where(tr14 != 0, 100 * dm_minus_14 / tr14, 0)
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    
    adx = np.zeros_like(dx)
    adx[27] = np.nanmean(dx[14:28])  # First ADX value
    for i in range(28, len(dx)):
        adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    adx_14 = adx
    adx_14[:27] = np.nan
    
    # Align ADX to 4h
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Calculate Donchian channels (20-period)
    donchian_high = np.zeros_like(high)
    donchian_low = np.zeros_like(low)
    donchian_mid = np.zeros_like(high)
    
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-19:i+1])
        donchian_low[i] = np.min(low[i-19:i+1])
        donchian_mid[i] = (donchian_high[i] + donchian_low[i]) / 2
    
    # Fill beginning with NaN
    donchian_high[:20] = np.nan
    donchian_low[:20] = np.nan
    donchian_mid[:20] = np.nan
    
    # Calculate 4h volume SMA(20)
    vol_sma_4h = np.zeros_like(volume)
    vol_sma_4h[:] = np.nan
    for i in range(20, n):
        vol_sma_4h[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = max(30, 28)  # need ADX and Donchian
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(adx_14_aligned[i]) or 
            np.isnan(vol_sma_4h[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_sma_val = vol_sma_4h[i]
        adx_val = adx_14_aligned[i]
        dc_high = donchian_high[i]
        dc_low = donchian_low[i]
        dc_mid = donchian_mid[i]
        
        if position == 0:
            # Long: Price breaks above Donchian high + volume > 2x SMA + ADX > 25
            if price > dc_high and close[i-1] <= dc_high and vol > 2.0 * vol_sma_val and adx_val > 25:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low + volume > 2x SMA + ADX > 25
            elif price < dc_low and close[i-1] >= dc_low and vol > 2.0 * vol_sma_val and adx_val > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price retests midpoint or breaks below low
            if price < dc_mid or price < dc_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price retests midpoint or breaks above high
            if price > dc_mid or price > dc_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_VolumeSpike_1dADX25"
timeframe = "4h"
leverage = 1.0