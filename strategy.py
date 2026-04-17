#!/usr/bin/env python3
"""
Hypothesis: 4h timeframe with Donchian(20) breakout + volume confirmation + 1d chop regime filter.
Long when price breaks above Donchian high(20) with volume > 1.3x 20-period average and chop > 61.8 (range).
Short when price breaks below Donchian low(20) with volume > 1.3x 20-period average and chop > 61.8 (range).
Exit when price reverts to Donchian midpoint or chop < 38.2 (trend).
Chop filter ensures we only trade in ranging markets where mean reversion works.
Target: 75-200 total trades over 4 years (19-50/year) to avoid fee drag. Uses discrete sizing 0.25.
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
    
    # Get 1d data for chop regime filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(14) for chop
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, tr2)])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d Donchian(20) for chop denominator
    highest_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    chop_denom = highest_1d - lowest_1d
    # Avoid division by zero
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)
    chop_1d = 100 * (np.log10(atr_1d * np.sqrt(20) / chop_denom) / np.log10(20))
    
    # Get 4h data for Donchian breakout
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate 4h Donchian(20)
    highest_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_mid_4h = (highest_4h + lowest_4h) / 2
    
    # Calculate 4h volume 20-period average
    vol_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align all to 4h timeframe (primary)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    highest_4h_aligned = align_htf_to_ltf(prices, df_4h, highest_4h)
    lowest_4h_aligned = align_htf_to_ltf(prices, df_4h, lowest_4h)
    donchian_mid_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid_4h)
    vol_ma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    volume_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_4h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 40  # need enough for Donchian(20) and chop calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(chop_1d_aligned[i]) or np.isnan(highest_4h_aligned[i]) or 
            np.isnan(lowest_4h_aligned[i]) or np.isnan(donchian_mid_4h_aligned[i]) or 
            np.isnan(vol_ma_20_4h_aligned[i]) or np.isnan(volume_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.3x 20-period average
        volume_confirmed = volume_4h_aligned[i] > 1.3 * vol_ma_20_4h_aligned[i]
        
        # Chop regime: > 61.8 = ranging (good for mean reversion), < 38.2 = trending
        chop_ranging = chop_1d_aligned[i] > 61.8
        chop_trending = chop_1d_aligned[i] < 38.2
        
        if position == 0:
            # Long: price breaks above Donchian high with volume and ranging market
            if (close[i] > highest_4h_aligned[i] and 
                volume_confirmed and 
                chop_ranging):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume and ranging market
            elif (close[i] < lowest_4h_aligned[i] and 
                  volume_confirmed and 
                  chop_ranging):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to Donchian midpoint OR market starts trending
            if (close[i] < donchian_mid_4h_aligned[i] or 
                chop_trending):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to Donchian midpoint OR market starts trending
            if (close[i] > donchian_mid_4h_aligned[i] or 
                chop_trending):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Volume_ChopRegime"
timeframe = "4h"
leverage = 1.0