#!/usr/bin/env python3
# Hypothesis: 6h Donchian(20) breakout with weekly pivot regime filter and 1d volume confirmation.
# Long when price breaks above 20-period Donchian high AND weekly pivot shows bullish bias (price above weekly pivot) AND 1d volume > 1.5x average.
# Short when price breaks below 20-period Donchian low AND weekly pivot shows bearish bias (price below weekly pivot) AND 1d volume > 1.5x average.
# Exit when price breaks opposite Donchian level (long exits on Donchian low break, short exits on Donchian high break).
# Uses 6h timeframe for lower frequency, Donchian for structure, weekly pivot for regime, 1d volume for confirmation.
# Target: 50-150 total trades over 4 years (12-37/year). Works in bull via breakout continuation, bear via faded rallies.

name = "6h_Donchian20_WeeklyPivot_1dVolume_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Donchian calculation
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate Donchian channels on 6h
    donchian_high_20 = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to primary timeframe
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_6h, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_6h, donchian_low_20)
    
    # Get weekly data for pivot calculation (regime filter)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (standard formula)
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    weekly_r1 = 2 * weekly_pivot - low_1w
    weekly_s1 = 2 * weekly_pivot - high_1w
    
    # Align weekly pivot to primary timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume average (20-period)
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_filter_1d = volume_1d > (1.5 * vol_ma_1d)
    volume_filter_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_filter_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_20_aligned[i]) or np.isnan(donchian_low_20_aligned[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(volume_filter_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian high AND price above weekly pivot (bullish regime) AND volume confirmation
            if close[i] > donchian_high_20_aligned[i] and close[i] > weekly_pivot_aligned[i] and volume_filter_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low AND price below weekly pivot (bearish regime) AND volume confirmation
            elif close[i] < donchian_low_20_aligned[i] and close[i] < weekly_pivot_aligned[i] and volume_filter_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian low (opposite breakout)
            if close[i] < donchian_low_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian high (opposite breakout)
            if close[i] > donchian_high_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals