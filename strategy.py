#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index (CHOP) regime filter + Donchian breakout + volume confirmation
# In trending markets (CHOP < 38.2), trade Donchian breakouts. In ranging markets (CHOP > 61.8), fade extremes.
# Works in both bull and bear: trend-following in trends, mean-reversion in ranges.
# Uses 1d CHOP for regime, 4h Donchian(20) for entries, volume for confirmation.
# Target: 20-50 trades/year to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Choppiness Index (14-period) on 1d
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # ATR (smoothed TR)
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of TR over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(tr_sum / (hh_1d - ll_1d)) / log10(14)
    # Avoid division by zero and log of zero
    diff = hh_1d - ll_1d
    diff = np.where(diff <= 0, 1e-10, diff)
    chop = 100 * np.log10(tr_sum / diff) / np.log10(14)
    chop = np.where(tr_sum <= 0, 50, chop)  # Neutral when no movement
    
    # Align CHOP to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate Donchian channels (20-period) on 4h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if any required data is NaN
        if (np.isnan(chop_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i])):
            continue
        
        chop_val = chop_aligned[i]
        
        # Trending market: CHOP < 38.2 -> trade Donchian breakouts
        if chop_val < 38.2:
            # Long: break above upper Donchian + volume confirmation
            if (close[i] > highest_high[i] and
                volume[i] > 1.5 * np.median(window := volume[max(0, i-20):i+1]) and
                position <= 0):
                position = 1
                signals[i] = base_size
            
            # Short: break below lower Donchian + volume confirmation
            elif (close[i] < lowest_low[i] and
                  volume[i] > 1.5 * np.median(window := volume[max(0, i-20):i+1]) and
                  position >= 0):
                position = -1
                signals[i] = -base_size
        
        # Ranging market: CHOP > 61.8 -> fade extremes (mean reversion)
        elif chop_val > 61.8:
            # Long: near lower Donchian (support) + volume confirmation
            if (close[i] <= lowest_low[i] * 1.001 and  # Within 0.1% of low
                volume[i] > 1.5 * np.median(window := volume[max(0, i-20):i+1]) and
                position <= 0):
                position = 1
                signals[i] = base_size
            
            # Short: near upper Donchian (resistance) + volume confirmation
            elif (close[i] >= highest_high[i] * 0.999 and  # Within 0.1% of high
                  volume[i] > 1.5 * np.median(window := volume[max(0, i-20):i+1]) and
                  position >= 0):
                position = -1
                signals[i] = -base_size
        
        # Exit conditions: opposite signal or chop moves to neutral zone
        elif position == 1 and (close[i] < lowest_low[i] or chop_val > 50):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > highest_high[i] or chop_val > 50):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_Chop_Donchian_Breakout_MeanReversion"
timeframe = "4h"
leverage = 1.0