#!/usr/bin/env python3
"""
Hypothesis: 12h timeframe with 1d Donchian(20) breakout + volume confirmation + choppiness regime filter.
Long when price breaks above 1d Donchian upper channel with volume confirmation and chop regime favors trending (CHOP < 61.8).
Short when price breaks below 1d Donchian lower channel with volume confirmation and chop regime favors trending (CHOP < 61.8).
Exit when price returns to the 1d Donchian midpoint.
Designed to capture institutional breakouts with volume confirmation while avoiding false signals in choppy/ranging markets.
Uses 12h for signal generation (lower frequency = less fee drag) and 1d Donchian/chop for structure.
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
    
    # Get 1d data for Donchian channel and choppiness calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian(20) channels
    lookback = 20
    upper_1d = pd.Series(high_1d).rolling(window=lookback, min_periods=lookback).max().values
    lower_1d = pd.Series(low_1d).rolling(window=lookback, min_periods=lookback).min().values
    mid_1d = (upper_1d + lower_1d) / 2.0
    
    # Calculate 1d choppiness index (CHOP) - values > 61.8 indicate choppy/ranging
    # CHOP = 100 * log10(sum(ATR(14)) / log10(highest_high - lowest_low) / lookback) / log10(lookback)
    tr1 = pd.Series(high_1d).rolling(window=14, min_periods=14).max() - pd.Series(low_1d).rolling(window=14, min_periods=14).min()
    tr2 = abs(pd.Series(high_1d) - pd.Series(close_1d).shift(1))
    tr3 = abs(pd.Series(low_1d) - pd.Series(close_1d).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.rolling(window=14, min_periods=14).mean().values
    highest_high = pd.Series(high_1d).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low_1d).rolling(window=lookback, min_periods=lookback).min().values
    chop_raw = 100 * np.log10(atr14.sum() / (highest_high - lowest_low)) / np.log10(lookback) if (highest_high - lowest_low) > 0 else 100
    # Simplified rolling CHOP calculation
    atr_sum = pd.Series(atr14).rolling(window=lookback, min_periods=lookback).sum().values
    hh_ll_diff = highest_high - lowest_low
    chop = 100 * np.log10(atr_sum / hh_ll_diff) / np.log10(lookback)
    chop = np.where(hh_ll_diff > 0, chop, 100.0)  # avoid division by zero
    
    # Calculate 12h volume 20-period average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 12h timeframe
    upper_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_1d)
    lower_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_1d)
    mid_1d_aligned = align_htf_to_ltf(prices, df_1d, mid_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for volume MA and indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_1d_aligned[i]) or 
            np.isnan(lower_1d_aligned[i]) or 
            np.isnan(mid_1d_aligned[i]) or 
            np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Regime filter: only trade when market is trending (CHOP < 61.8)
        trending_regime = chop_aligned[i] < 61.8
        
        if position == 0:
            # Long: price breaks above 1d Donchian upper with volume and trending regime
            if (close[i] > upper_1d_aligned[i] and 
                volume_confirmed and 
                trending_regime):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Donchian lower with volume and trending regime
            elif (close[i] < lower_1d_aligned[i] and 
                  volume_confirmed and 
                  trending_regime):
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