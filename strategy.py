#!/usr/bin/env python3
# 6h_1w_donchian_weekly_pivot_volume_v1
# Strategy: 6h Donchian(20) breakout with weekly pivot direction and volume confirmation
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: Donchian breakouts capture trend momentum. Weekly pivot (from 1w) provides
# directional bias: long above weekly pivot, short below. Volume confirms breakout strength.
# Works in bull by riding uptrends, in bear by catching breakdowns below pivot.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_donchian_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly pivot point and support/resistance levels
    # Using typical price: (H + L + C) / 3
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    pivot = typical_price.values
    # Calculate R1, S1, R2, S2
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    r1 = 2 * pivot - low_w
    s1 = 2 * pivot - high_w
    r2 = pivot + (high_w - low_w)
    s2 = pivot - (high_w - low_w)
    
    # Use weekly pivot as trend filter: above pivot = bullish bias, below = bearish
    weekly_bias = pivot > (high_w + low_w + df_1w['close'].values) / 3  # Actually just pivot > typical price? No, pivot IS typical price
    # Correction: weekly bias based on current price vs weekly pivot
    # We'll compute this inside loop using current price and last weekly pivot
    
    # Align weekly pivot to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    
    # 6h Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation (20-period average)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)  # Volume spike filter
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(lookback, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_avg[i]) or np.isnan(pivot_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Current price and levels
        price_now = close[i]
        donchian_high = highest_high[i]
        donchian_low = lowest_low[i]
        weekly_pivot = pivot_aligned[i]
        
        # Determine weekly bias: price above/below weekly pivot
        bias_long = price_now > weekly_pivot
        bias_short = price_now < weekly_pivot
        
        # Breakout conditions with volume confirmation
        breakout_long = (price_now > donchian_high) and vol_spike[i]
        breakout_short = (price_now < donchian_low) and vol_spike[i]
        
        # Exit conditions: opposite Donchian breakout or loss of bias
        exit_long = position == 1 and (
            (price_now < donchian_low) or  # Price breaks below Donchian low
            not bias_long  # Lost bullish bias
        )
        exit_short = position == -1 and (
            (price_now > donchian_high) or  # Price breaks above Donchian high
            not bias_short  # Lost bearish bias
        )
        
        # Trading logic: trade breakouts in direction of weekly bias
        if breakout_long and bias_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif breakout_short and bias_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals