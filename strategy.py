#!/usr/bin/env python3
"""
Hypothesis: 12h timeframe with 1d Donchian(20) breakout + volume confirmation + chop regime filter.
Long when price breaks above 1d Donchian upper channel with volume confirmation and chop regime > 61.8 (ranging market for mean reversion exit logic).
Short when price breaks below 1d Donchian lower channel with volume confirmation and chop regime > 61.8.
Exit when price returns to the 1d Donchian midpoint.
Uses 1d timeframe for structure (reduces noise) and 12h for execution timing.
Chop regime filter avoids whipsaws in strong trends where breakouts fail.
Designed for low-frequency, high-conviction trades in both bull and bear markets.
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
    
    # Calculate 1d Donchian(20) channels
    lookback = 20
    upper_1d = pd.Series(high_1d).rolling(window=lookback, min_periods=lookback).max().values
    lower_1d = pd.Series(low_1d).rolling(window=lookback, min_periods=lookback).min().values
    mid_1d = (upper_1d + lower_1d) / 2.0
    
    # Calculate 1d ATR(14) for chop regime
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close[:-1])
    tr3 = np.abs(low_1d[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d chop regime: CHOP = 100 * log10(sum(atr,14) / (max(high,20)-min(low,20))) / log10(14)
    max_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    min_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    sum_atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    denominator = max_high_20 - min_low_20
    chop_raw = 100 * np.log10(sum_atr_14 / denominator) / np.log10(14)
    chop_regime = np.where(denominator > 0, chop_raw, 50.0)  # default to 50 if invalid
    
    # Calculate 12h volume 20-period average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 12h timeframe
    upper_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_1d)
    lower_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_1d)
    mid_1d_aligned = align_htf_to_ltf(prices, df_1d, mid_1d)
    chop_regime_aligned = align_htf_to_ltf(prices, df_1d, chop_regime)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for ATR and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_1d_aligned[i]) or 
            np.isnan(lower_1d_aligned[i]) or 
            np.isnan(mid_1d_aligned[i]) or 
            np.isnan(chop_regime_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        # Chop regime filter: only trade in ranging markets (CHOP > 61.8)
        chop_filter = chop_regime_aligned[i] > 61.8
        
        if position == 0:
            # Long: price breaks above 1d Donchian upper with volume and chop regime
            if (close[i] > upper_1d_aligned[i] and 
                volume_confirmed and 
                chop_filter):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Donchian lower with volume and chop regime
            elif (close[i] < lower_1d_aligned[i] and 
                  volume_confirmed and 
                  chop_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to or below 1d Donchian midpoint
            if close[i] <= mid_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to or above 1d Donchian midpoint
            if close[i] >= mid_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1dDonchian20_Breakout_Volume_ChopFilter"
timeframe = "12h"
leverage = 1.0