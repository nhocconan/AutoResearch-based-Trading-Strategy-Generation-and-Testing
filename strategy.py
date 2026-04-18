#!/usr/bin/env python3
"""
4h Donchian Breakout + Volume Spike + ADX Trend Filter
Strategy: Long when price breaks above Donchian(20) upper band with volume spike and ADX>25.
          Short when price breaks below Donchian(20) lower band with volume spike and ADX>25.
          Use daily EMA200 as trend filter to avoid counter-trend trades in bear markets.
          Designed for low trade frequency with clear breakout edge in both bull and bear markets.
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
    
    # Get daily data for EMA200 trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily EMA200 for trend filter
    daily_close = df_1d['close'].values
    ema_200_1d = pd.Series(daily_close).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate ADX(14) on 4h data
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    atr = np.zeros(n)
    dm_plus_smooth = np.zeros(n)
    dm_minus_smooth = np.zeros(n)
    
    # Initial values
    atr[13] = np.mean(tr[1:14])  # 14-period average
    dm_plus_smooth[13] = np.mean(dm_plus[1:14])
    dm_minus_smooth[13] = np.mean(dm_minus[1:14])
    
    # Wilder's smoothing
    for i in range(14, n):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
        dm_plus_smooth[i] = (dm_plus_smooth[i-1] * 13 + dm_plus[i]) / 14
        dm_minus_smooth[i] = (dm_minus_smooth[i-1] * 13 + dm_minus[i]) / 14
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = np.zeros(n)
    adx[27] = np.mean(dx[14:28])  # First ADX value
    for i in range(28, n):
        adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Donchian Channel (20-period)
    donch_high = np.zeros(n)
    donch_low = np.zeros(n)
    for i in range(20, n):
        donch_high[i] = np.max(high[i-20:i])
        donch_low[i] = np.min(low[i-20:i])
    
    # Volume spike detection (2x 20-period average)
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 30  # need enough history for ADX and Donchian
    
    for i in range(start_idx, n):
        if (np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(adx[i]) or
            np.isnan(donch_high[i]) or
            np.isnan(donch_low[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        donch_high_level = donch_high[i]
        donch_low_level = donch_low[i]
        adx_value = adx[i]
        ema_200 = ema_200_1d_aligned[i]
        
        if position == 0:
            # Long: break above Donchian high with volume spike, ADX>25, and above daily EMA200
            if (price > donch_high_level and volume_spike[i] and adx_value > 25 and price > ema_200):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with volume spike, ADX>25, and below daily EMA200
            elif (price < donch_low_level and volume_spike[i] and adx_value > 25 and price < ema_200):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position management
            signals[i] = 0.25
            # Exit: price breaks below Donchian low or below daily EMA200
            if price < donch_low_level or price < ema_200:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.25
            # Exit: price breaks above Donchian high or above daily EMA200
            if price > donch_high_level or price > ema_200:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_VolumeSpike_ADXFilter_EMA200"
timeframe = "4h"
leverage = 1.0