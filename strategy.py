#!/usr/bin/env python3
# 1d_donchian_1w_trend_volume_v1
# Hypothesis: Daily Donchian breakout (20-period) aligned with weekly trend (price above/below weekly midpoint) and volume confirmation (>1.5x 20-period average).
# Weekly trend filter avoids counter-trend trades in bear markets. Volume confirms institutional participation.
# Discrete position sizing (±0.25) to minimize fee churn. Target: 30-100 total trades over 4 years (7-25/year).
# Works in both bull and bear markets by requiring alignment with weekly structure.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w HTF data for weekly Donchian and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly Donchian channels (20-period)
    highest_high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    weekly_midpoint = (highest_high_20 + lowest_low_20) / 2
    
    # Align weekly Donchian to 1d timeframe
    highest_high_20_aligned = align_htf_to_ltf(prices, df_1w, highest_high_20)
    lowest_low_20_aligned = align_htf_to_ltf(prices, df_1w, lowest_low_20)
    weekly_midpoint_aligned = align_htf_to_ltf(prices, df_1w, weekly_midpoint)
    
    # 1d Donchian channels (20-period)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high_20_aligned[i]) or np.isnan(lowest_low_20_aligned[i]) or
            np.isnan(weekly_midpoint_aligned[i]) or np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below weekly midpoint OR below 1d Donchian low
            if close[i] < weekly_midpoint_aligned[i] or close[i] < lowest_low_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above weekly midpoint OR above 1d Donchian high
            if close[i] > weekly_midpoint_aligned[i] or close[i] > highest_high_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need volume confirmation
            volume_confirmed = volume[i] > 1.5 * volume_ma[i]
            
            if volume_confirmed:
                # Long: price breaks above 1d Donchian high WITH weekly bullish bias
                if close[i] > highest_high_20[i] and close[i] > weekly_midpoint_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below 1d Donchian low WITH weekly bearish bias
                elif close[i] < lowest_low_20[i] and close[i] < weekly_midpoint_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals