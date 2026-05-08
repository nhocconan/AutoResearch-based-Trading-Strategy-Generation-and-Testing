#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter + 12h Donchian breakout with volume confirmation.
# Long when price breaks above 12h Donchian upper band (20) AND 4h Choppiness Index > 61.8 (ranging market) AND volume > 1.5x 20-period average.
# Short when price breaks below 12h Donchian lower band (20) AND 4h Choppiness Index > 61.8 AND volume > 1.5x 20-period average.
# Exit when price crosses back inside 12h Donchian channel (between upper and lower bands).
# This strategy trades range-bound markets with clear breakout signals, using Choppiness Index to avoid trending markets where breakouts fail.
# The 12h Donchian provides structural breakout levels, volume confirms participation, and Choppiness filters for optimal ranging conditions.
# Target: 20-50 total trades over 4 years (5-12.5/year) to minimize fee drag.

name = "4h_Choppiness_Donchian_12h_Volume"
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
    
    # 12h data for Donchian calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period high/low) from 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Donchian upper band (20-period high)
    donchian_high_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    # Donchian lower band (20-period low)
    donchian_low_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high_12h)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low_12h)
    
    # 4h data for Choppiness Index calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range calculation
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    
    # ATR(14) - smoothed true range
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Sum of ATR over 14 periods
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    highest_high_14 = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(sum_atr_14 / (highest_high_14 - lowest_low_14)) / log10(14)
    # Avoid division by zero
    range_14 = highest_high_14 - lowest_low_14
    chop_raw = np.zeros_like(close_4h)
    mask = range_14 > 0
    chop_raw[mask] = 100 * np.log10(sum_atr_14[mask] / range_14[mask]) / np.log10(14)
    
    # Align Choppiness Index to 4h timeframe (already on 4h, but ensure alignment)
    chop_4h = chop_raw  # Already calculated on 4h data
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(chop_4h[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high, chop > 61.8 (ranging), volume filter
            long_cond = (close[i] > donchian_high_aligned[i]) and (chop_4h[i] > 61.8) and volume_filter[i]
            # Short conditions: price breaks below Donchian low, chop > 61.8 (ranging), volume filter
            short_cond = (close[i] < donchian_low_aligned[i]) and (chop_4h[i] > 61.8) and volume_filter[i]
            
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