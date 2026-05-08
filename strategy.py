#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Choppiness Index regime filter combined with 1d Donchian breakout.
# Long when: price breaks above Donchian(20) upper band AND Choppiness Index > 61.8 (ranging market)
# Short when: price breaks below Donchian(20) lower band AND Choppiness Index > 61.8
# Exit when price crosses back inside Donchian channel.
# This strategy trades mean reversion in ranging markets using Donchian breakouts as entry signals.
# Choppiness Index > 61.8 indicates ranging conditions where breakouts often fail and reverse.
# Target: 50-150 total trades over 4 years (12-37/year) with controlled frequency.

name = "12h_Chop_Donchian_MeanRev"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1d data for Donchian calculation
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
    
    # Align Donchian levels to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # 1d data for Choppiness Index calculation
    # Choppiness Index: 100 * log10(sum(ATR over n) / (log10(highest_high - lowest_low) * n))
    tr1 = np.maximum(high_1d[1:], low_1d[:-1]) - np.minimum(low_1d[1:], high_1d[:-1])
    tr1 = np.concatenate([[high_1d[0] - low_1d[0]], tr1])
    
    atr_sum = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_hl = highest_high - lowest_low
    
    # Avoid division by zero
    chop_raw = np.where(range_hl > 0, atr_sum / (range_hl * 14), 100)
    chop = 100 * np.log10(chop_raw)
    
    # Align Choppiness Index to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long/short conditions: Donchian breakout AND ranging market (Chop > 61.8)
            long_cond = (close[i] > donchian_high_aligned[i]) and (chop_aligned[i] > 61.8)
            short_cond = (close[i] < donchian_low_aligned[i]) and (chop_aligned[i] > 61.8)
            
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