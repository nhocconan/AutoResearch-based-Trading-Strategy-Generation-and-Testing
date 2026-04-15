#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h 1-week Donchian Channel Breakout with Volume Confirmation and 1-day ATR Filter
# Uses weekly Donchian channels (20-period high/low) as support/resistance. 
# Breakouts are traded only when confirmed by above-average volume and 1-day ATR < 20-day ATR (low volatility environment).
# Works in bull markets (breakouts up) and bear markets (breakouts down). Target: 50-150 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Load 1d data for ATR filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 20-period Donchian channels on 1w
    high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 12h timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1w, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1w, low_20)
    
    # Calculate 14-period ATR on 1d
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period ATR on 1d
    atr_20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Align ATRs to 12h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_20_aligned = align_htf_to_ltf(prices, df_1d, atr_20)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size (25% of capital)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or
            np.isnan(atr_14_aligned[i]) or np.isnan(atr_20_aligned[i])):
            continue
        
        # Long entry: price breaks above weekly Donchian high + volume confirmation + low volatility (ATR14 < ATR20)
        if (close[i] > high_20_aligned[i] and
            volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
            atr_14_aligned[i] < atr_20_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below weekly Donchian low + volume confirmation + low volatility
        elif (close[i] < low_20_aligned[i] and
              volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
              atr_14_aligned[i] < atr_20_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse breakout or high volatility (ATR14 >= ATR20)
        elif position == 1 and (close[i] < low_20_aligned[i] or atr_14_aligned[i] >= atr_20_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > high_20_aligned[i] or atr_14_aligned[i] >= atr_20_aligned[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "12h_1w_Donchian_Breakout_Volume_ATR"
timeframe = "12h"
leverage = 1.0