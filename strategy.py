#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily Bollinger Bands (20, 2.5)
    close_1d = df_1d['close'].values
    sma_20 = np.full(len(close_1d), np.nan)
    std_20 = np.full(len(close_1d), np.nan)
    
    for i in range(20, len(close_1d)):
        sma_20[i] = np.mean(close_1d[i-20:i])
        std_20[i] = np.std(close_1d[i-20:i])
    
    upper_band = sma_20 + (std_20 * 2.5)
    lower_band = sma_20 - (std_20 * 2.5)
    
    # Align Bollinger Bands to 4h timeframe
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    sma_20_aligned = align_htf_to_ltf(prices, df_1d, sma_20)
    
    # Calculate 4h Bollinger Bands for entry timing
    sma_20_4h = np.full(n, np.nan)
    std_20_4h = np.full(n, np.nan)
    
    for i in range(20, n):
        sma_20_4h[i] = np.mean(close[i-20:i])
        std_20_4h[i] = np.std(close[i-20:i])
    
    upper_band_4h = sma_20_4h + (std_20_4h * 2.0)
    lower_band_4h = sma_20_4h - (std_20_4h * 2.0)
    
    # Volume filter
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        if (np.isnan(upper_band_aligned[i]) or
            np.isnan(lower_band_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 0 else 0
        
        # Volume confirmation: > 2.0x average volume
        volume_confirmation = vol_ratio > 2.0
        
        if position == 0:
            # Long: price touches daily lower Bollinger Band with volume confirmation
            if volume_confirmation and price <= lower_band_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price touches daily upper Bollinger Band with volume confirmation
            elif volume_confirmation and price >= upper_band_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price returns to daily SMA or opposite band touched
            if price >= sma_20_aligned[i] or price >= upper_band_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to daily SMA or opposite band touched
            if price <= sma_20_aligned[i] or price <= lower_band_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1D_Bollinger_Bands_Touch_Reversion"
timeframe = "4h"
leverage = 1.0