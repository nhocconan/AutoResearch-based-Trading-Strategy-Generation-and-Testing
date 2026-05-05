#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and volume confirmation
# Long when: price breaks above Donchian upper band (20-period high), close > weekly pivot R1, volume > 1.5x 20-period average
# Short when: price breaks below Donchian lower band (20-period low), close < weekly pivot S1, volume > 1.5x 20-period average
# Exit when price returns to Donchian midpoint (mean reversion) or opposite breakout
# Uses weekly pivot for structural bias (bull/bear regime) and Donchian for breakout timing
# Timeframe: 6h, HTF: 1w. Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_Donchian20_Breakout_WeeklyPivot_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Calculate Donchian channels on 6h
    if len(high) >= 20:
        donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_mid = (donchian_upper + donchian_lower) / 2
    else:
        donchian_upper = np.full(n, np.nan)
        donchian_lower = np.full(n, np.nan)
        donchian_mid = np.full(n, np.nan)
    
    # Volume confirmation on 6h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get weekly data ONCE before loop for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (standard formula)
    # Pivot = (H + L + C)/3
    # R1 = 2*P - L, S1 = 2*P - H
    if len(high_1w) >= 1:
        pivot_1w = (high_1w + low_1w + close_1w) / 3
        r1_1w = 2 * pivot_1w - low_1w
        s1_1w = 2 * pivot_1w - high_1w
    else:
        pivot_1w = np.full(len(close_1w), np.nan)
        r1_1w = np.full(len(close_1w), np.nan)
        s1_1w = np.full(len(close_1w), np.nan)
    
    # Align weekly pivot levels to 6h timeframe
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_mid[i]) or 
            np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper, above weekly R1, volume filter
            if (close[i] > donchian_upper[i] and 
                open_price[i] <= donchian_upper[i] and  # Ensure breakout happens on this bar
                close[i] > r1_1w_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower, below weekly S1, volume filter
            elif (close[i] < donchian_lower[i] and 
                  open_price[i] >= donchian_lower[i] and  # Ensure breakdown happens on this bar
                  close[i] < s1_1w_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below Donchian midpoint (mean reversion) or breaks below lower (reversal)
            if close[i] < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above Donchian midpoint (mean reversion) or breaks above upper (reversal)
            if close[i] > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals