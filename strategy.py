#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + volume confirmation + 1d chop regime filter.
Long when price breaks above Donchian(20) high with volume > 1.8x 20-period average and 1d chop > 61.8 (range).
Short when price breaks below Donchian(20) low with volume > 1.8x 20-period average and 1d chop > 61.8 (range).
Exit when price returns to Donchian(20) midpoint or chop < 38.2 (trend).
Chop regime filter ensures we only trade breakouts in ranging markets, avoiding whipsaws in strong trends.
Target: 75-200 total trades over 4 years (19-50/year). Uses discrete sizing 0.25.
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
    
    # Calculate Donchian(20) on 4h
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (highest_high + lowest_low) / 2
    
    # Get 1d data for chop regime filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Chopiness Index(14) on 1d
    def calculate_chop(high_arr, low_arr, close_arr, window=14):
        atr = pd.Series(np.maximum(np.maximum(high_arr - low_arr, 
                                             np.abs(high_arr - np.roll(close_arr, 1))), 
                                 np.abs(low_arr - np.roll(close_arr, 1))))
        atr.iloc[0] = high_arr[0] - low_arr[0]  # first ATR
        atr_sum = atr.rolling(window=window, min_periods=window).sum()
        highest_high_window = pd.Series(high_arr).rolling(window=window, min_periods=window).max()
        lowest_low_window = pd.Series(low_arr).rolling(window=window, min_periods=window).min()
        chop = 100 * np.log10(atr_sum / (highest_high_window - lowest_low_window)) / np.log10(window)
        return chop.values
    
    chop_1d = calculate_chop(high_1d, low_1d, close_1d, 14)
    
    # Calculate 1d volume 20-period average
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all to 4h
    highest_high_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high}), highest_high)
    lowest_low_aligned = align_htf_to_ltf(prices, pd.DataFrame({'low': low}), lowest_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), donchian_mid)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(20, 20)  # Donchian lookback and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(highest_high_aligned[i]) or np.isnan(lowest_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(chop_1d_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.8x 20-period average
        volume_confirmed = volume_1d_aligned[i] > 1.8 * vol_ma_20_1d_aligned[i]
        
        # Chop regime filter: only trade when chop > 61.8 (ranging market)
        chop_filter = chop_1d_aligned[i] > 61.8
        
        if position == 0:
            # Long: price breaks above Donchian high with volume and chop filter
            if (close[i] > highest_high_aligned[i] and 
                volume_confirmed and 
                chop_filter):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume and chop filter
            elif (close[i] < lowest_low_aligned[i] and 
                  volume_confirmed and 
                  chop_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to Donchian midpoint OR chop < 38.2 (trend emerging)
            if (close[i] < donchian_mid_aligned[i] or 
                chop_1d_aligned[i] < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to Donchian midpoint OR chop < 38.2 (trend emerging)
            if (close[i] > donchian_mid_aligned[i] or 
                chop_1d_aligned[i] < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Volume_ChopRegime"
timeframe = "4h"
leverage = 1.0