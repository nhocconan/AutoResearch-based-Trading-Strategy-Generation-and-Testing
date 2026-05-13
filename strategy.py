#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with volume confirmation and choppiness regime filter.
# Long when price breaks above Donchian(20) high AND volume > 1.5x average AND choppy market (CHOP > 61.8).
# Short when price breaks below Donchian(20) low AND volume > 1.5x average AND choppy market (CHOP > 61.8).
# Exit when price crosses Donchian(10) midpoint OR choppiness regime ends (CHOP < 38.2).
# Uses 4h timeframe for lower frequency, Donchian for structure, volume for confirmation, chop for regime.
# Target: 75-200 total trades over 4 years (19-50/year). Works in bull via breakouts, bear via faded rallies in chop.

name = "4h_Donchian20_Volume_Chop_v2"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels on 4h data
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Donchian(20) for breakout
    donchian_high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_mid_10 = (pd.Series(high_4h).rolling(window=10, min_periods=10).max().values + 
                       pd.Series(low_4h).rolling(window=10, min_periods=10).min().values) / 2
    
    # Donchian(20) aligned to LTF
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_20)
    donchian_mid_10_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid_10)
    
    # Volume filter: current 4h volume > 1.5x 20-period average
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    volume_filter = volume_4h > (1.5 * vol_ma_4h)
    volume_filter_aligned = align_htf_to_ltf(prices, df_4h, volume_filter)
    
    # Choppiness Index (CHOP) on 1d data for regime filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # ATR(14) and sum of true ranges
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Choppiness Index: CHOP = 100 * log10(sum_tr_14 / (atr_14 * 14)) / log10(14)
    chop = 100 * np.log10(sum_tr_14 / (atr_14 * 14)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Regime filters: choppy market (CHOP > 61.8) for mean reversion bias
    chop_filter = chop_aligned > 61.8
    trending_filter = chop_aligned < 38.2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_20_aligned[i]) or np.isnan(donchian_low_20_aligned[i]) or
            np.isnan(donchian_mid_10_aligned[i]) or np.isnan(vol_ma_4h_aligned[i]) or
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian(20) high AND volume confirmation AND choppy market
            if (close[i] > donchian_high_20_aligned[i] and 
                volume_filter_aligned[i] and 
                chop_filter[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian(20) low AND volume confirmation AND choppy market
            elif (close[i] < donchian_low_20_aligned[i] and 
                  volume_filter_aligned[i] and 
                  chop_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses Donchian(10) midpoint OR market becomes trending
            if (close[i] < donchian_mid_10_aligned[i] or 
                trending_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses Donchian(10) midpoint OR market becomes trending
            if (close[i] > donchian_mid_10_aligned[i] or 
                trending_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals