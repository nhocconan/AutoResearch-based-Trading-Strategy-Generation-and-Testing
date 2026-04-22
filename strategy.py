#!/usr/bin/env python3
"""
Hypothesis: Daily Donchian breakout with 1-week ADX trend filter and volume confirmation.
Long when price breaks above Donchian(20) high, weekly ADX > 25, and volume > 1.5x 20-day average.
Short when price breaks below Donchian(20) low, weekly ADX > 25, and volume > 1.5x 20-day average.
Exit when price crosses opposite Donchian boundary or ADX drops below 20.
Designed for low trade frequency by requiring strong breakouts with trend and volume confirmation.
Works in trending markets (both bull and bear) by filtering out weak moves and chop.
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
    
    # Load 1-week data for ADX trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX(14) on weekly data
    # True Range
    tr1 = np.abs(high_1w - low_1w)
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
    
    # Smooth TR, DM+ and DM- using Wilder's smoothing (alpha = 1/period)
    atr = np.zeros_like(tr)
    dm_plus_smooth = np.zeros_like(dm_plus)
    dm_minus_smooth = np.zeros_like(dm_minus)
    
    # Initial values (simple average of first 14 periods)
    if len(tr) >= 14:
        atr[13] = np.mean(tr[1:14])
        dm_plus_smooth[13] = np.mean(dm_plus[1:14])
        dm_minus_smooth[13] = np.mean(dm_minus[1:14])
        
        # Wilder's smoothing for rest
        for i in range(14, len(tr)):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * 13 + dm_plus[i]) / 14
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * 13 + dm_minus[i]) / 14
    
    # Avoid division by zero
    atr_safe = np.where(atr == 0, 1e-10, atr)
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / atr_safe
    di_minus = 100 * dm_minus_smooth / atr_safe
    
    # DX and ADX
    dx = np.zeros_like(di_plus)
    dx_sum = np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10) * 100
    adx = np.zeros_like(dx)
    
    # Initial ADX (average of first 14 DX values)
    if len(dx_sum) >= 14:
        adx[13] = np.mean(dx_sum[1:14])
        # Wilder's smoothing for ADX
        for i in range(14, len(dx_sum)):
            adx[i] = (adx[i-1] * 13 + dx_sum[i]) / 14
    
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Donchian Channel (20-period) on daily data
    high_20 = np.zeros_like(high)
    low_20 = np.zeros_like(low)
    
    for i in range(20, len(high)):
        high_20[i] = np.max(high[i-19:i+1])
        low_20[i] = np.min(low[i-19:i+1])
    
    # Volume average (20-period)
    vol_avg = np.zeros_like(volume)
    for i in range(20, len(volume)):
        vol_avg[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(vol_avg[i]) or np.isnan(adx_aligned[i]) or
            vol_avg[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high, ADX > 25, volume > 1.5x average
            if (close[i] > high_20[i] and 
                adx_aligned[i] > 25 and 
                volume[i] > 1.5 * vol_avg[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low, ADX > 25, volume > 1.5x average
            elif (close[i] < low_20[i] and 
                  adx_aligned[i] > 25 and 
                  volume[i] > 1.5 * vol_avg[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price breaks below Donchian low OR ADX drops below 20
                if (close[i] < low_20[i] or 
                    adx_aligned[i] < 20):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price breaks above Donchian high OR ADX drops below 20
                if (close[i] > high_20[i] or 
                    adx_aligned[i] < 20):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Donchian20_1wADX25_VolumeFilter"
timeframe = "1d"
leverage = 1.0