#!/usr/bin/env python3

# Hypothesis: 12h timeframe with 1d Donchian channel breakout and 1d ATR filter.
# Uses 1-day Donchian channels (20-period) for breakout entries and 1-day ATR for volatility filtering.
# The Donchian channel provides clear breakout levels that work in trending markets,
# while the ATR filter ensures we only trade during sufficient volatility, avoiding choppy periods.
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25.

name = "12h_Donchian20_1dATR_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Donchian channel and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1-day Donchian channel (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Upper band: highest high over past 20 days
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low over past 20 days
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Calculate 1-day ATR (14-period) for volatility filter
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close, 1)[len(df_1d):] if len(df_1d) > len(high_1d) else np.abs(high_1d - np.roll(df_1d['close'].values, 1)))
    tr3 = np.abs(low_1d - np.roll(close, 1)[len(df_1d):] if len(df_1d) > len(low_1d) else np.abs(low_1d - np.roll(df_1d['close'].values, 1)))
    # Handle the first element for TR calculation
    tr_temp = np.maximum(tr1, tr2)
    tr_temp = np.maximum(tr_temp, tr3)
    tr_temp[0] = tr1[0]  # First TR is just high-low
    atr_14 = pd.Series(tr_temp).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Calculate average ATR for filtering (50-period average)
    avg_atr = pd.Series(atr_14_aligned).rolling(window=50, min_periods=50).mean().values
    
    # Volatility filter: current ATR > 0.5 x average ATR (ensures sufficient volatility)
    vol_filter = atr_14_aligned > (0.5 * avg_atr)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Need enough data for indicators (20+14+50 for Donchian, ATR, and average ATR)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(atr_14_aligned[i]) or np.isnan(avg_atr[i]) or
            np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high + volatility filter
            if close[i] > donchian_high_aligned[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + volatility filter
            elif close[i] < donchian_low_aligned[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to Donchian low or volatility drops
            if close[i] < donchian_low_aligned[i] or not vol_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to Donchian high or volatility drops
            if close[i] > donchian_high_aligned[i] or not vol_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals