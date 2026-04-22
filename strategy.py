#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and volume confirmation
    # Works in both bull and bear markets: breakouts from price channels capture directional moves
    # Weekly pivot provides long-term bias, volume surge confirms breakout strength
    
    # Load weekly data once
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (using prior week)
    pivot = (high_1w[:-1] + low_1w[:-1] + close_1w[:-1]) / 3
    r1 = 2 * pivot - low_1w[:-1]
    s1 = 2 * pivot - high_1w[:-1]
    r2 = pivot + (high_1w[:-1] - low_1w[:-1])
    s2 = pivot - (high_1w[:-1] - low_1w[:-1])
    # Shift to align with current week (pivot based on prior week)
    pivot = np.concatenate([np.array([np.nan]), pivot])
    r1 = np.concatenate([np.array([np.nan]), r1])
    s1 = np.concatenate([np.array([np.nan]), s1])
    r2 = np.concatenate([np.array([np.nan]), r2])
    s2 = np.concatenate([np.array([np.nan]), s2])
    
    # Align weekly pivot to 6h
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # 6h Donchian channel (20-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    vol = prices['volume'].values
    
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter (20-period surge)
    vol_ma20 = pd.Series(vol).rolling(window=20, min_periods=20).mean().values
    vol_surge = vol > 2.0 * vol_ma20
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Donchian breakout above upper band with volume surge AND price above weekly R1 (bullish bias)
            if close[i] > donchian_high[i] and vol_surge[i] and close[i] > r1_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout below lower band with volume surge AND price below weekly S1 (bearish bias)
            elif close[i] < donchian_low[i] and vol_surge[i] and close[i] < s1_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to Donchian middle or opposite band touch
            donchian_mid = (donchian_high[i] + donchian_low[i]) / 2
            if position == 1:
                if close[i] < donchian_mid:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > donchian_mid:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Donchian_Breakout_WeeklyPivot_R1S1_VolumeSurge_v1"
timeframe = "6h"
leverage = 1.0