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
    
    # Load weekly data (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 20-period high and low for Donchian channel (weekly)
    donchian_high_20 = np.full_like(close_1w, np.nan)
    donchian_low_20 = np.full_like(close_1w, np.nan)
    
    if len(close_1w) >= 20:
        for i in range(19, len(close_1w)):
            donchian_high_20[i] = np.max(high_1w[i-19:i+1])
            donchian_low_20[i] = np.min(low_1w[i-19:i+1])
    
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_20)
    
    # Calculate 14-period ADX for trend strength (weekly)
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w),
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)),
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    
    atr_14 = np.full_like(close_1w, np.nan)
    dm_plus_14 = np.full_like(close_1w, np.nan)
    dm_minus_14 = np.full_like(close_1w, np.nan)
    
    if len(close_1w) >= 14:
        atr_14[13] = np.mean(tr[1:15])
        dm_plus_14[13] = np.mean(dm_plus[1:15])
        dm_minus_14[13] = np.mean(dm_minus[1:15])
        
        for i in range(15, len(close_1w)):
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
            dm_plus_14[i] = (dm_plus_14[i-1] * 13 + dm_plus[i]) / 14
            dm_minus_14[i] = (dm_minus_14[i-1] * 13 + dm_minus[i]) / 14
    
    di_plus = np.full_like(close_1w, np.nan)
    di_minus = np.full_like(close_1w, np.nan)
    
    for i in range(14, len(close_1w)):
        if atr_14[i] > 0:
            di_plus[i] = (dm_plus_14[i] / atr_14[i]) * 100
            di_minus[i] = (dm_minus_14[i] / atr_14[i]) * 100
    
    dx = np.full_like(close_1w, np.nan)
    adx_14 = np.full_like(close_1w, np.nan)
    
    for i in range(14, len(close_1w)):
        if di_plus[i] + di_minus[i] > 0:
            dx[i] = (np.abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])) * 100
    
    if len(close_1w) >= 28:
        adx_14[27] = np.mean(dx[14:28])
        for i in range(28, len(close_1w)):
            adx_14[i] = (adx_14[i-1] * 13 + dx[i]) / 14
    
    adx_14_aligned = align_htf_to_ltf(prices, df_1w, adx_14)
    
    # Calculate 20-period volume average (weekly)
    vol_ma_20 = np.full_like(df_1w['volume'].values, np.nan)
    vol_1w = df_1w['volume'].values
    if len(vol_1w) >= 20:
        for i in range(19, len(vol_1w)):
            vol_ma_20[i] = np.mean(vol_1w[i-19:i+1])
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20)
    
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
        
        # Volume ratio: current daily volume vs 20-period weekly average volume
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

name = "1d_1w_Donchian20_ADX25_Volume"
timeframe = "1d"
leverage = 1.0