#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Choppiness Index regime filter with 1-day Donchian breakout
# Long when price > 1-day Donchian high AND chop > 61.8 (trending regime)
# Short when price < 1-day Donchian low AND chop > 61.8 (trending regime)
# Choppiness Index identifies trending vs ranging markets; only trade in trending regimes
# Donchian breakout provides clear entry/exit signals with built-in trend following
# Targets 75-200 total trades over 4 years (19-50/year) by requiring both trend and breakout

name = "4h_ChopRegime_DonchianBreakout"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get daily data once for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1-day Donchian channels (20-period high/low)
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    donchian_high = pd.Series(daily_high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(daily_low).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Calculate Choppiness Index (14-period)
    atr = np.zeros(n)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr / (highest_high - lowest_low)) / np.log10(14)
    # Handle division by zero and invalid values
    chop = np.where((highest_high - lowest_low) != 0, chop, 50.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        dh = donchian_high_aligned[i]
        dl = donchian_low_aligned[i]
        price = close[i]
        chop_val = chop[i]
        
        if position == 0:
            # Enter long: price > 1-day Donchian high AND trending regime (chop > 61.8)
            if price > dh and chop_val > 61.8:
                signals[i] = 0.25
                position = 1
            # Enter short: price < 1-day Donchian low AND trending regime (chop > 61.8)
            elif price < dl and chop_val > 61.8:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price < 1-day Donchian low OR chop < 38.2 (ranging regime)
            if price < dl or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price > 1-day Donchian high OR chop < 38.2 (ranging regime)
            if price > dh or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals