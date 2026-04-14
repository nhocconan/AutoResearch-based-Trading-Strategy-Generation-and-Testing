#!/usr/bin/env python3
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
    
    # Load daily data (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-period high and low for Donchian channel (daily)
    donchian_high_20 = np.full_like(close_1d, np.nan)
    donchian_low_20 = np.full_like(close_1d, np.nan)
    
    if len(close_1d) >= 20:
        for i in range(19, len(close_1d)):
            donchian_high_20[i] = np.max(high_1d[i-19:i+1])
            donchian_low_20[i] = np.min(low_1d[i-19:i+1])
    
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    
    # Calculate 14-period ADX for trend strength (daily)
    # Calculate True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Calculate +DM and -DM
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    
    # Smooth TR, DM+, DM- using Wilder's smoothing (14-period)
    atr_14 = np.full_like(close_1d, np.nan)
    dm_plus_14 = np.full_like(close_1d, np.nan)
    dm_minus_14 = np.full_like(close_1d, np.nan)
    
    if len(close_1d) >= 14:
        atr_14[13] = np.mean(tr[1:15])
        dm_plus_14[13] = np.mean(dm_plus[1:15])
        dm_minus_14[13] = np.mean(dm_minus[1:15])
        
        for i in range(15, len(close_1d)):
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
            dm_plus_14[i] = (dm_plus_14[i-1] * 13 + dm_plus[i]) / 14
            dm_minus_14[i] = (dm_minus_14[i-1] * 13 + dm_minus[i]) / 14
    
    # Calculate DI+ and DI-
    di_plus = np.full_like(close_1d, np.nan)
    di_minus = np.full_like(close_1d, np.nan)
    
    for i in range(14, len(close_1d)):
        if atr_14[i] > 0:
            di_plus[i] = (dm_plus_14[i] / atr_14[i]) * 100
            di_minus[i] = (dm_minus_14[i] / atr_14[i]) * 100
    
    # Calculate DX and ADX
    dx = np.full_like(close_1d, np.nan)
    adx_14 = np.full_like(close_1d, np.nan)
    
    for i in range(14, len(close_1d)):
        if di_plus[i] + di_minus[i] > 0:
            dx[i] = (np.abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])) * 100
    
    if len(close_1d) >= 28:
        adx_14[27] = np.mean(dx[14:28])
        for i in range(28, len(close_1d)):
            adx_14[i] = (adx_14[i-1] * 13 + dx[i]) / 14
    
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Calculate 20-period volume average (daily)
    vol_ma_20 = np.full_like(volume_1d, np.nan)
    if len(volume_1d) >= 20:
        for i in range(19, len(volume_1d)):
            vol_ma_20[i] = np.mean(volume_1d[i-19:i+1])
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high_20_aligned[i]) or 
            np.isnan(donchian_low_20_aligned[i]) or 
            np.isnan(adx_14_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 12h volume vs 20-period daily average volume
        if vol_ma_20_aligned[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20_aligned[i]
        
        if position == 0:
            # Long: Price breaks above Donchian high + ADX > 25 + volume surge
            if (close[i] > donchian_high_20_aligned[i] and
                adx_14_aligned[i] > 25 and
                volume_ratio > 2.0):
                position = 1
                signals[i] = position_size
            # Short: Price breaks below Donchian low + ADX > 25 + volume surge
            elif (close[i] < donchian_low_20_aligned[i] and
                  adx_14_aligned[i] > 25 and
                  volume_ratio > 2.0):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price breaks below Donchian low OR ADX < 20
            if (close[i] < donchian_low_20_aligned[i] or 
                adx_14_aligned[i] < 20):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price breaks above Donchian high OR ADX < 20
            if (close[i] > donchian_high_20_aligned[i] or 
                adx_14_aligned[i] < 20):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Donchian20_ADX25_Volume"
timeframe = "12h"
leverage = 1.0