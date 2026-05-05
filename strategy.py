#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h HMA(21) trend filter and volume confirmation (2.0x)
# Long when price breaks above Donchian upper AND price > 12h HMA21 AND volume > 2.0x 20-period average
# Short when price breaks below Donchian lower AND price < 12h HMA21 AND volume > 2.0x 20-period average
# Exit when price reverts to Donchian midpoint OR 12h HMA21 filter reverses
# Uses Donchian channels for clear trend structure + volume confirmation to reduce false signals
# 12h HMA21 provides higher timeframe trend filter effective in both bull and bear markets
# Designed for 75-200 total trades over 4 years (19-50/year) to minimize fee drag
# Timeframe: 4h (primary), HTF: 12h

name = "4h_Donchian20_12hHMA21_VolumeSpike_2.0x"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for HMA21
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # Calculate Donchian(20) on 4h
    if len(high) >= 20:
        donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_mid = (donchian_upper + donchian_lower) / 2.0
    else:
        donchian_upper = np.full(n, np.nan)
        donchian_lower = np.full(n, np.nan)
        donchian_mid = np.full(n, np.nan)
    
    # Calculate 12h HMA(21)
    if len(close_12h) >= 21:
        # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        half_n = 21 // 2
        sqrt_n = int(np.sqrt(21))
        wma_half = pd.Series(close_12h).rolling(window=half_n, min_periods=half_n).mean().values
        wma_full = pd.Series(close_12h).rolling(window=21, min_periods=21).mean().values
        raw_hma = 2 * wma_half - wma_full
        hma_21_12h = pd.Series(raw_hma).rolling(window=sqrt_n, min_periods=sqrt_n).mean().values
    else:
        hma_21_12h = np.full(len(close_12h), np.nan)
    
    # Align HTF indicators to 4h timeframe
    hma_21_aligned = align_htf_to_ltf(prices, df_12h, hma_21_12h)
    
    # Volume confirmation on 4h (threshold: 2.0x for optimal frequency)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (2.0 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(hma_21_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper AND price > HMA21 AND volume spike
            if (close[i] > donchian_upper[i] and 
                close[i] > hma_21_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower AND price < HMA21 AND volume spike
            elif (close[i] < donchian_lower[i] and 
                  close[i] < hma_21_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price reverts to midpoint OR price < HMA21 (trend weakening)
            if close[i] < donchian_mid[i] or close[i] < hma_21_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reverts to midpoint OR price > HMA21 (trend weakening)
            if close[i] > donchian_mid[i] or close[i] > hma_21_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals