#!/usr/bin/env python3
"""
exp_6455_6h_donchian20_1d_pivot_vol_v1
Hypothesis: 6h Donchian(20) breakout with daily pivot confirmation and volume filter.
Works in both bull and bear markets by trading breakouts with institutional reference points (pivots).
Daily pivots provide static support/resistance that price respects across regimes.
Volume confirmation ensures breakouts have conviction.
Target: 50-150 trades over 4 years (12-37/year).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_6455_6h_donchian20_1d_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    signals = np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return signals
    
    # Calculate daily pivot points (standard formula)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    r2 = pivot + (high_1d - low_1d)
    s2 = pivot - (high_1d - low_1d)
    r3 = high_1d + 2 * (pivot - low_1d)
    s3 = low_1d - 2 * (high_1d - pivot)
    
    # Align pivot levels to 6h timeframe (shifted by 1 for completed daily bars only)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate Donchian channels on 6h
    lookback = 20
    high_roll = prices['high'].rolling(window=lookback, min_periods=lookback).max()
    low_roll = prices['low'].rolling(window=lookback, min_periods=lookback).min()
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean()
    vol_ratio = prices['volume'] / vol_ma
    
    for i in range(lookback, n):
        # Skip if volume data not ready
        if pd.isna(vol_ratio.iloc[i]):
            continue
            
        # Volume filter: need strong volume for breakout
        if vol_ratio.iloc[i] < 1.5:
            continue
        
        # Get current price and levels
        close_price = prices['close'].iloc[i]
        high_price = prices['high'].iloc[i]
        low_price = prices['low'].iloc[i]
        
        # Get pivot levels for this bar (from previous completed daily bar)
        pivot_level = pivot_aligned[i]
        r1_level = r1_aligned[i]
        s1_level = s1_aligned[i]
        r2_level = r2_aligned[i]
        s2_level = s2_aligned[i]
        r3_level = r3_aligned[i]
        s3_level = s3_aligned[i]
        
        # Skip if pivot levels not ready
        if pd.isna(pivot_level) or pd.isna(r1_level) or pd.isna(s1_level):
            continue
        
        # Donchian breakout conditions
        donchian_high = high_roll.iloc[i]
        donchian_low = low_roll.iloc[i]
        
        # Long: price breaks above Donchian high with volume, above daily pivot
        if (high_price > donchian_high and 
            close_price > pivot_level and
            close_price > r1_level):
            signals[i] = 0.25
        
        # Short: price breaks below Donchian low with volume, below daily pivot
        elif (low_price < donchian_low and 
              close_price < pivot_level and
              close_price < s1_level):
            signals[i] = -0.25
        
        # Exit conditions: reverse signal or stoploss proximity
        # Exit long if price breaks below Donchian low or below S1
        elif signals[i-1] > 0 and (low_price < donchian_low or close_price < s1_level):
            signals[i] = 0.0
        
        # Exit short if price breaks above Donchian high or above R1
        elif signals[i-1] < 0 and (high_price > donchian_high or close_price > r1_level):
            signals[i] = 0.0
    
    return signals