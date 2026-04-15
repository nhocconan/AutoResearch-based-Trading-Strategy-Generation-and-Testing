#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter + Donchian breakout + volume confirmation
# Uses 1d Choppiness Index to identify trending vs ranging markets.
# In trending markets (CHOP < 38.2): trade Donchian(20) breakouts with volume confirmation.
# In ranging markets (CHOP > 61.8): fade Donchian breaks at Bollinger Bands (20,2).
# Works in bull markets (breakouts up) and bear markets (breakouts down).
# Target: 80-160 total trades over 4 years (20-40/year).
# Timeframe: 4h, HTF: 1d

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for Choppiness Index and Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Choppiness Index (14-period)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    chop = 100 * np.log10(atr_1d.sum() / (highest_high - lowest_low)) / np.log10(14)
    # Handle division by zero and invalid values
    chop = np.where((highest_high - lowest_low) > 0, chop, 50.0)
    chop = np.where(np.isnan(chop), 50.0, chop)
    
    # Calculate 1d Bollinger Bands (20,2)
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    
    # Calculate 4h Donchian channels (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Align 1d indicators to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(chop_aligned[i]) or np.isnan(upper_bb_aligned[i]) or 
            np.isnan(lower_bb_aligned[i]) or np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i])):
            continue
        
        chop_val = chop_aligned[i]
        
        # Trending market regime: CHOP < 38.2
        if chop_val < 38.2:
            # Long: Donchian breakout above + volume confirmation
            if (close[i] > highest_20[i] and 
                volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
                position <= 0):
                position = 1
                signals[i] = base_size
            
            # Short: Donchian breakout below + volume confirmation
            elif (close[i] < lowest_20[i] and 
                  volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
                  position >= 0):
                position = -1
                signals[i] = -base_size
        
        # Ranging market regime: CHOP > 61.8
        elif chop_val > 61.8:
            # Fade Donchian breaks at Bollinger Bands
            # Short when price breaks above Donchian high and touches upper BB
            if (close[i] > highest_20[i] and 
                close[i] >= upper_bb_aligned[i] and
                position >= 0):
                position = -1
                signals[i] = -base_size
            
            # Long when price breaks below Donchian low and touches lower BB
            elif (close[i] < lowest_20[i] and 
                  close[i] <= lower_bb_aligned[i] and
                  position <= 0):
                position = 1
                signals[i] = base_size
        
        # Exit conditions
        if position == 1 and (close[i] < lowest_20[i] or chop_val > 61.8):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > highest_20[i] or chop_val > 61.8):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_Chop_Donchian_BB_Switch"
timeframe = "4h"
leverage = 1.0