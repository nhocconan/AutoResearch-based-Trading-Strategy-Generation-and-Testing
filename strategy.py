#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Choppiness Index regime filter with 1d Donchian breakout and volume confirmation
# Choppiness Index (CHOP) > 61.8 = ranging market (mean reversion), CHOP < 38.2 = trending market (trend follow)
# In ranging markets (CHOP > 61.8): fade 1d Donchian extremes (R4/S4) with volume confirmation
# In trending markets (CHOP < 38.2): breakout continuation of 1d Donchian extremes (R4/S4) with volume confirmation
# This adaptive regime approach works in both bull and bear markets by switching logic based on market state
# Discrete sizing 0.25 targets 50-150 total trades over 4 years (12-37/year) for 6h timeframe

name = "6h_ChopRegime_1dDonchian_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d Donchian channels (20-period) from prior completed 1d bar
    highest_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    highest_20_shifted = np.roll(highest_20, 1)
    lowest_20_shifted = np.roll(lowest_20, 1)
    highest_20_shifted[0] = np.nan
    lowest_20_shifted[0] = np.nan
    highest_20_aligned = align_htf_to_ltf(prices, df_1d, highest_20_shifted)
    lowest_20_aligned = align_htf_to_ltf(prices, df_1d, lowest_20_shifted)
    
    # Calculate Choppiness Index (14-period) on 6h data
    # CHOP = 100 * log10(sum(ATR(1)) / (max(high) - min(low))) / log10(14)
    tr1 = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr1[0] = 0  # First bar has no prior close
    atr1 = tr1  # ATR(1) is just true range
    sum_atr1 = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * (np.log10(sum_atr1) - np.log10(max_high - min_low)) / np.log10(14)
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(chop[i]) or np.isnan(highest_20_aligned[i]) or 
            np.isnan(lowest_20_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Determine regime: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
            if chop[i] > 61.8:  # Ranging market - mean reversion
                # Fade 1d Donchian extremes: short at R4, long at S4
                if close[i] >= highest_20_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                    signals[i] = -0.25  # Short at resistance
                    position = -1
                elif close[i] <= lowest_20_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                    signals[i] = 0.25   # Long at support
                    position = 1
            elif chop[i] < 38.2:  # Trending market - breakout continuation
                # Breakout continuation: long at R4, short at S4
                if close[i] > highest_20_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                    signals[i] = 0.25   # Long breakout
                    position = 1
                elif close[i] < lowest_20_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                    signals[i] = -0.25  # Short breakdown
                    position = -1
        elif position == 1:
            # Exit long: price retreats to midpoint of 1d Donchian channel
            midpoint = (highest_20_aligned[i] + lowest_20_aligned[i]) / 2
            if close[i] <= midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price advances to midpoint of 1d Donchian channel
            midpoint = (highest_20_aligned[i] + lowest_20_aligned[i]) / 2
            if close[i] >= midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals