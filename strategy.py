#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeTrend_HMA
Hypothesis: 4h Donchian(20) breakouts with volume confirmation and HMA(21) trend filter capture strong momentum moves. Works in both bull and bear markets by requiring volume surge and trend alignment, reducing false breakouts. Target: 20-50 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data for Donchian and HMA
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Donchian(20) channels
    upper = np.full_like(close_4h, np.nan)
    lower = np.full_like(close_4h, np.nan)
    for i in range(20, len(close_4h)):
        upper[i] = np.max(high_4h[i-20:i])
        lower[i] = np.min(low_4h[i-20:i])
    
    # HMA(21) for trend filter
    def wma(arr, n):
        if len(arr) < n:
            return np.full_like(arr, np.nan)
        weights = np.arange(1, n + 1)
        return np.convolve(arr, weights/weights.sum(), mode='valid')
    
    def hma(arr, n):
        half_n = n // 2
        sqrt_n = int(np.sqrt(n))
        wma_half = wma(arr, half_n)
        wma_full = wma(arr, n)
        wma2_half = 2 * wma_half
        diff = wma2_half - wma_full
        if len(diff) < sqrt_n:
            return np.full_like(arr, np.nan)
        return wma(diff, sqrt_n)
    
    hma_21 = hma(close_4h, 21)
    
    # Align to lower timeframe (1h base, but we use 4h as primary)
    upper_aligned = align_htf_to_ltf(prices, df_4h, upper)
    lower_aligned = align_htf_to_ltf(prices, df_4h, lower)
    hma_21_aligned = align_htf_to_ltf(prices, df_4h, hma_21)
    
    # 1h data for volume and price
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 2.0x 24-period average (more selective)
    volume_avg = np.zeros_like(volume)
    for i in range(len(volume)):
        if i >= 24:
            volume_avg[i] = np.mean(volume[i-24:i])
        else:
            volume_avg[i] = np.mean(volume[:i+1]) if i > 0 else volume[i]
    volume_filter = volume > (2.0 * volume_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # start after warmup
        # Skip if NaN in critical values
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(hma_21_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        upper_chan = upper_aligned[i]
        lower_chan = lower_aligned[i]
        hma_val = hma_21_aligned[i]
        vol_ok = volume_filter[i]
        
        if position == 0:
            # Long: break above upper Donchian with volume and above HMA (uptrend)
            if price > upper_chan and vol_ok and price > hma_val:
                signals[i] = 0.30
                position = 1
            # Short: break below lower Donchian with volume and below HMA (downtrend)
            elif price < lower_chan and vol_ok and price < hma_val:
                signals[i] = -0.30
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below HMA or breaks lower Donchian (reversal)
            if price < hma_val or price < lower_chan:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Short exit: price crosses above HMA or breaks upper Donchian (reversal)
            if price > hma_val or price > upper_chan:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Donchian20_Breakout_VolumeTrend_HMA"
timeframe = "4h"
leverage = 1.0