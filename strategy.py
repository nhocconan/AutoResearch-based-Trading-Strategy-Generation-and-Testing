#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour strategy using 1-day Choppiness Index (CHOP) regime filter with
# Donchian channel breakout and volume confirmation. CHOP > 61.8 indicates ranging market
# (mean-reversion), CHOP < 38.2 indicates trending (breakout continuation).
# Long when: price > Donchian Upper(20) + CHOP < 38.2 + volume > 1.5x average
# Short when: price < Donchian Lower(20) + CHOP < 38.2 + volume > 1.5x average
# Exit when: price crosses Donchian midline or CHOP > 61.8 (range regime)
# Designed for low trade frequency (target: 50-150 total trades over 4 years) to minimize fee drag.
# Works in bull markets (breakout continuations) and bear markets (breakdown continuations).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Choppiness Index calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1-day Choppiness Index (CHOP) with period 14
    # TR = max(high-low, abs(high-previous_close), abs(low-previous_close))
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    
    # Sum of true ranges over period
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over period
    hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(atr_sum / (hh - ll)) / log10(14)
    # Avoid division by zero
    range_hl = hh - ll
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    chop = 100 * np.log10(atr_sum / range_hl) / np.log10(14)
    
    # Align Choppiness Index to 12h timeframe (wait for 1d bar to close)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Donchian channels (20-period) on 12h data
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2.0
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(chop_aligned[i]) or np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or np.isnan(donch_mid[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trending regime: CHOP < 38.2
        trending = chop_aligned[i] < 38.2
        
        # Long breakout: price breaks above Donchian High with volume in trending regime
        if trending and close[i] > donch_high[i] and volume_filter[i]:
            signals[i] = 0.25
            position = 1
        # Short breakdown: price breaks below Donchian Low with volume in trending regime
        elif trending and close[i] < donch_low[i] and volume_filter[i]:
            signals[i] = -0.25
            position = -1
        # Exit conditions
        elif position == 1 and (close[i] <= donch_mid[i] or chop_aligned[i] > 61.8):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] >= donch_mid[i] or chop_aligned[i] > 61.8):
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_Donchian20_CHOP14_VolumeFilter"
timeframe = "12h"
leverage = 1.0