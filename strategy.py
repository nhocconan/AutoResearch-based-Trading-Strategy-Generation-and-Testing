#!/usr/bin/env python3
"""
Hypothesis: 12h timeframe with 1d Donchian channel breakout + volume confirmation + chop regime filter.
Long when price breaks above 1d Donchian upper (20) with volume > 1.5x 20-period average and chop > 61.8 (range).
Short when price breaks below 1d Donchian lower (20) with volume confirmation and chop > 61.8.
Uses Donchian for structure, volume for breakout validity, chop to avoid false breakouts in strong trends.
Designed to capture mean reversion in ranging markets (chop high) while avoiding trend-following whipsaws.
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
    
    # Get 1d data for Donchian channel calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian channels (20-period)
    # Upper = max(high_1d, 20)
    # Lower = min(low_1d, 20)
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    upper_1d = high_series.rolling(window=20, min_periods=20).max().values
    lower_1d = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h ATR (14) for choppy market detection
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr2 = np.maximum(tr1, np.abs(low[1:] - close[:-1]))
    tr = np.concatenate([[np.nan], tr2])  # first tr is nan
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 12h TRUE RANGE for chop calculation (using 14-period)
    # Chop = 100 * log10(sum(atr,14) / (n * log10(atr_max - atr_min))) / log10(n)
    # Simplified: Chop = 100 * log10(atr_sum / (atr_max - atr_min)) / log10(period)
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    atr_max = pd.Series(atr).rolling(window=14, min_periods=14).max().values
    atr_min = pd.Series(atr).rolling(window=14, min_periods=14).min().values
    range_14 = atr_max - atr_min
    # Avoid division by zero
    chop = np.where((range_14 > 0) & (atr_sum > 0), 
                    100 * np.log10(atr_sum / range_14) / np.log10(14), 
                    50.0)  # neutral when undefined
    
    # Calculate 12h volume 20-period average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d Donchian levels to 12h timeframe
    upper_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_1d)
    lower_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for Donchian(20) and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_1d_aligned[i]) or 
            np.isnan(lower_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or 
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        # Chop regime filter: chop > 61.8 indicates ranging market (mean reversion favorable)
        chop_filter = chop[i] > 61.8
        
        if position == 0:
            # Long: price breaks above 1d Donchian upper with volume and chop > 61.8 (range)
            if (close[i] > upper_1d_aligned[i] and 
                volume_confirmed and 
                chop_filter):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Donchian lower with volume and chop > 61.8 (range)
            elif (close[i] < lower_1d_aligned[i] and 
                  volume_confirmed and 
                  chop_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below 1d Donchian lower (opposite side)
            if close[i] < lower_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above 1d Donchian upper (opposite side)
            if close[i] > upper_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1dDonchian20_Breakout_Volume_ChopFilter"
timeframe = "12h"
leverage = 1.0