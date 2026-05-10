#!/usr/bin/env python3
# 6h_Donchian20_WeeklyPivot_Direction_Volume
# Hypothesis: Breakout from daily Donchian channel (20-period) with weekly pivot direction filter and volume confirmation.
# Weekly pivot determines trend direction (above/below weekly pivot), Donchian breakout provides entry, volume confirms strength.
# Works in both bull/bear markets: weekly pivot adapts to long-term trend, Donchian captures breakouts, volume avoids false signals.
# Target: 20-40 trades/year to stay within fee-efficient range.

name = "6h_Donchian20_WeeklyPivot_Direction_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data for Donchian and volume calculations
    df_1d = get_htf_data(prices, '1d')
    # Get weekly data for pivot point
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 20 or len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate daily Donchian channel (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate weekly pivot point (standard: (H+L+C)/3)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    
    # Align indicators to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Get price and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Donchian (20)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(weekly_pivot_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above Donchian high, above weekly pivot, with volume
            if close[i] > donchian_high_aligned[i] and close[i] > weekly_pivot_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low, below weekly pivot, with volume
            elif close[i] < donchian_low_aligned[i] and close[i] < weekly_pivot_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price back below Donchian low or below weekly pivot
            if close[i] < donchian_low_aligned[i] or close[i] < weekly_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price back above Donchian high or above weekly pivot
            if close[i] > donchian_high_aligned[i] or close[i] > weekly_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals