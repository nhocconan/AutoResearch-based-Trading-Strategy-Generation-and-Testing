#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Donchian20_WeeklyPivotDir_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for pivot and Donchian
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Weekly OHLC for pivot calculation
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    open_1w = df_1w['open'].values
    
    # Weekly pivot point: P = (H + L + C) / 3
    pivot = (high_1w + low_1w + close_1w) / 3.0
    # Weekly support/resistance levels
    r1 = 2 * pivot - low_1w
    s1 = 2 * pivot - high_1w
    r2 = pivot + (high_1w - low_1w)
    s2 = pivot - (high_1w - low_1w)
    
    # Use previous week's levels to avoid look-ahead
    pivot_prev = np.roll(pivot, 1)
    r1_prev = np.roll(r1, 1)
    s1_prev = np.roll(s1, 1)
    r2_prev = np.roll(r2, 1)
    s2_prev = np.roll(s2, 1)
    pivot_prev[0] = np.nan
    r1_prev[0] = np.nan
    s1_prev[0] = np.nan
    r2_prev[0] = np.nan
    s2_prev[0] = np.nan
    
    # Align weekly pivot levels to 6h timeframe
    pivot_6h = align_htf_to_ltf(prices, df_1w, pivot_prev)
    r1_6h = align_htf_to_ltf(prices, df_1w, r1_prev)
    s1_6h = align_htf_to_ltf(prices, df_1w, s1_prev)
    r2_6h = align_htf_to_ltf(prices, df_1w, r2_prev)
    s2_6h = align_htf_to_ltf(prices, df_1w, s2_prev)
    
    # Daily Donchian channel (20-period high/low)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian channels
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian to 6h timeframe
    donchian_high_6h = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_6h = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Volume filter: volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(pivot_6h[i]) or np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or np.isnan(r2_6h[i]) or np.isnan(s2_6h[i]) or
            np.isnan(donchian_high_6h[i]) or np.isnan(donchian_low_6h[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above weekly R1 with volume spike and above Donchian high
            if (price > r1_6h[i] and vol_spike[i] and price > donchian_high_6h[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S1 with volume spike and below Donchian low
            elif (price < s1_6h[i] and vol_spike[i] and price < donchian_low_6h[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below weekly pivot (mean reversion to pivot)
            if price < pivot_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above weekly pivot (mean reversion to pivot)
            if price > pivot_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals