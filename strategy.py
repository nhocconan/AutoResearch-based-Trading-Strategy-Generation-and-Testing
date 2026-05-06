#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Donchian breakout with volume confirmation
# Weekly Donchian(20) breakouts capture major trend changes with minimal trades
# Volume > 1.5x 20-day average confirms institutional participation
# Works in bull markets (breakouts up) and bear markets (breakouts down)
# Target: 10-30 trades over 4 years (2-7/year) with 0.30 position sizing

name = "1d_Donchian20_WeeklyBreakout_Volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate weekly Donchian channels ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly Donchian(20) - highest high and lowest low of past 20 weeks
    high_20 = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min().values
    
    # Align weekly levels to daily timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1w, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1w, low_20)
    
    # Volume confirmation: >1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above weekly Donchian high with volume
            if close[i] > high_20_aligned[i] and volume_filter[i]:
                signals[i] = 0.30
                position = 1
            # Short entry: price breaks below weekly Donchian low with volume
            elif close[i] < low_20_aligned[i] and volume_filter[i]:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price breaks below weekly Donchian low (trend reversal)
            if close[i] < low_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price breaks above weekly Donchian high (trend reversal)
            if close[i] > high_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals