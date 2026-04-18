#!/usr/bin/env python3
"""
4h Donchian Breakout + Volume Spike + 1d ADX Trend Filter
Long when price breaks above Donchian(20) high with volume spike and ADX > 25.
Short when price breaks below Donchian(20) low with volume spike and ADX > 25.
Uses 1d ADX as higher timeframe trend filter to ensure trades align with strong trends.
Designed for low trade frequency with clear trend-following edge.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for ADX trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d ADX(14) for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    atr = np.zeros_like(tr)
    dm_plus_smooth = np.zeros_like(dm_plus)
    dm_minus_smooth = np.zeros_like(dm_minus)
    
    # Initial values
    atr[13] = np.mean(tr[1:14])
    dm_plus_smooth[13] = np.mean(dm_plus[1:14])
    dm_minus_smooth[13] = np.mean(dm_minus[1:14])
    
    # Wilder's smoothing
    for i in range(14, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
        dm_plus_smooth[i] = (dm_plus_smooth[i-1] * 13 + dm_plus[i]) / 14
        dm_minus_smooth[i] = (dm_minus_smooth[i-1] * 13 + dm_minus[i]) / 14
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = np.zeros_like(dx)
    adx[27] = np.mean(dx[14:28])  # First ADX value after 2*period
    
    for i in range(28, len(dx)):
        adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    adx_1d = adx
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike detection (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 30  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(adx_1d_aligned[i]) or 
            np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: strong trend (ADX > 25)
        strong_trend = adx_1d_aligned[i] > 25
        
        price = close[i]
        
        if position == 0:
            # Long: break above Donchian high, volume spike, strong trend
            if (price > donch_high[i] and volume_spike[i] and strong_trend):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low, volume spike, strong trend
            elif (price < donch_low[i] and volume_spike[i] and strong_trend):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position management
            signals[i] = 0.25
            # Exit: break below Donchian low or weak trend
            if (price < donch_low[i]) or (adx_1d_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.25
            # Exit: break above Donchian high or weak trend
            if (price > donch_high[i]) or (adx_1d_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian_20_VolumeSpike_1dADX25"
timeframe = "4h"
leverage = 1.0