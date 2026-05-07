#!/usr/bin/env python3
"""
1d_1w_ChannelBreakout_TrendFilter
Hypothesis: Weekly Donchian channel breakouts with daily price position and volume confirmation capture major trend moves while avoiding false signals. The weekly channel provides strong support/resistance levels, and daily price above/below the weekly midpoint confirms trend alignment. Volume filters ensure breakout conviction. Low frequency via 1d timeframe and strict entry criteria reduces fee drag. Works in bull markets (breakouts) and bear markets (breakdowns).
Target: 30-100 total trades over 4 years.
"""
name = "1d_1w_ChannelBreakout_TrendFilter"
timeframe = "1d"
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
    
    # Weekly Donchian Channel (20-week period)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian high/low
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align to daily timeframe (wait for weekly close)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Weekly midpoint for trend filter
    weekly_midpoint = (donchian_high_aligned + donchian_low_aligned) / 2
    
    # Daily volume filter: volume > 1.5 * 20-day average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Need weekly Donchian and volume data
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(weekly_midpoint[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above weekly Donchian high + above weekly midpoint + volume
            if (close[i] > donchian_high_aligned[i] and 
                close[i] > weekly_midpoint[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian low + below weekly midpoint + volume
            elif (close[i] < donchian_low_aligned[i] and 
                  close[i] < weekly_midpoint[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price returns to weekly midpoint (mean reversion to weekly average)
            if position == 1:
                if close[i] <= weekly_midpoint[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] >= weekly_midpoint[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals