#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with volume confirmation and choppiness regime filter.
# Long when price breaks above Donchian(20) high AND volume > 1.5x average AND chop > 61.8 (range regime).
# Short when price breaks below Donchian(20) low AND volume > 1.5x average AND chop > 61.8 (range regime).
# Exit on opposite Donchian(10) break or chop < 38.2 (trend regime).
# Uses 4h timeframe for lower frequency, Donchian for structure, volume for confirmation, chop for regime.
# Target: 75-200 total trades over 4 years (19-50/year). Works in bull via breakout continuation, bear via faded rallies in range.

name = "4h_Donchian20_Volume_Chop_v1"
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
    
    # Get 4h data for Donchian and chop calculation
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate Donchian(20) on 4h
    donchian_high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_high_10 = pd.Series(high_4h).rolling(window=10, min_periods=10).max().values
    donchian_low_10 = pd.Series(low_4h).rolling(window=10, min_periods=10).min().values
    
    # Calculate Choppiness Index(14) on 4h
    atr_4h = pd.Series(high_4h - low_4h).rolling(window=14, min_periods=14).mean().values
    highest_high_14 = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    chop_denom = np.log10((highest_high_14 - lowest_low_14) / atr_4h) * np.sqrt(14)
    chop = 100 * (np.log10(chop_denom) / np.log10(14))
    # Handle division by zero and invalid values
    chop = np.where((highest_high_14 - lowest_low_14) > 0, chop, 50.0)
    
    # Volume filter: current 4h volume > 1.5x 20-period average
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_filter_4h = volume_4h > (1.5 * vol_ma_4h)
    
    # Align HTF indicators to LTF
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_20)
    donchian_high_10_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_10)
    donchian_low_10_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_10)
    chop_aligned = align_htf_to_ltf(prices, df_4h, chop)
    volume_filter_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_filter_4h.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_20_aligned[i]) or np.isnan(donchian_low_20_aligned[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(volume_filter_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian(20) high AND volume confirmation AND chop > 61.8 (range regime)
            if close[i] > donchian_high_20_aligned[i] and volume_filter_4h_aligned[i] > 0.5 and chop_aligned[i] > 61.8:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian(20) low AND volume confirmation AND chop > 61.8 (range regime)
            elif close[i] < donchian_low_20_aligned[i] and volume_filter_4h_aligned[i] > 0.5 and chop_aligned[i] > 61.8:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian(10) low OR chop < 38.2 (trend regime)
            if close[i] < donchian_low_10_aligned[i] or chop_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian(10) high OR chop < 38.2 (trend regime)
            if close[i] > donchian_high_10_aligned[i] or chop_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals