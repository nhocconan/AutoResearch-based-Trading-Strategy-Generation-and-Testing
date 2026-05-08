#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter with 1d Donchian breakout and volume confirmation.
# Long when price breaks above 1d Donchian upper band (20) AND 14-period Choppiness Index > 61.8 (range regime) AND volume > 1.5x 20-period average.
# Short when price breaks below 1d Donchian lower band (20) AND 14-period Choppiness Index > 61.8 (range regime) AND volume > 1.5x 20-period average.
# Exit when price crosses back inside the 1d Donchian channel.
# This strategy targets mean-reversion in ranging markets while avoiding trending conditions, which should work in both bull and bear markets.
# The Choppiness Index filter ensures we only trade in ranging markets (chop > 61.8), reducing false breakouts.
# Donchian breakouts provide clear entry levels, and volume confirmation ensures institutional participation.
# Target: 20-60 total trades over 4 years (5-15/year).

name = "4h_Chop_Donchian_20_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Donchian calculation and Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period high/low) from 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian upper band (20-period high)
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Donchian lower band (20-period low)
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Calculate Choppiness Index (14-period) from 1d data
    # CHOP = 100 * log10(sum(ATR14) / (max(high14) - min(low14))) / log10(14)
    tr1 = high_1d[1:] - low_1d[:-1]
    tr2 = np.abs(high_1d[1:] - df_1d['close'].values[:-1])
    tr3 = np.abs(low_1d[1:] - df_1d['close'].values[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    max_high14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range14 = max_high14 - min_low14
    range14 = np.where(range14 == 0, 1e-10, range14)
    
    chop = 100 * np.log10(atr14 / range14) / np.log10(14)
    chop = np.where(np.isnan(chop), 100, chop)  # Default to high chop when undefined
    
    # Align Choppiness Index to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # 14-period chop > 61.8 indicates ranging market (mean reversion regime)
    chop_filter = chop_aligned > 61.8
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Sufficient warmup for Donchian and chop
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high, chop > 61.8 (range), volume filter
            long_cond = (close[i] > donchian_high_aligned[i]) and chop_filter[i] and volume_filter[i]
            # Short conditions: price breaks below Donchian low, chop > 61.8 (range), volume filter
            short_cond = (close[i] < donchian_low_aligned[i]) and chop_filter[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below Donchian low
            if close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above Donchian high
            if close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals