#!/usr/bin/env python3
# 6h_donchian_1w_pivot_volume_v1
# Hypothesis: 6h strategy using weekly Donchian breakout (20-period) aligned with 1d pivot direction (H3/L3) and volume confirmation.
# Weekly Donchian provides structural breakout signals; 1d Camarilla H3/L3 acts as intraday support/resistance for continuation.
# Volume > 1.5x 20-period average confirms institutional participation.
# Designed for both bull and bear markets by requiring alignment with weekly trend (price above/below weekly midpoint).
# Discrete position sizing (±0.25) to minimize fee churn. Target: 75-200 total trades over 4 years (19-50/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_1w_pivot_volume_v1"
timeframe = "6h"
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
    
    # Align weekly Donchian to 6h timeframe
    highest_high_20_aligned = align_htf_to_ltf(prices, df_1w, highest_high_20)
    lowest_low_20_aligned = align_htf_to_ltf(prices, df_1w, lowest_low_20)
    weekly_midpoint_aligned = align_htf_to_ltf(prices, df_1w, weekly_midpoint)
    
    # 1d HTF data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d Camarilla pivot levels (based on previous day's range)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d[0] = np.nan
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    
    pivot_point = (prev_high_1d + prev_low_1d + prev_close_1d) / 3
    range_1d = prev_high_1d - prev_low_1d
    
    # Camarilla levels: H3, L3 (strongest intraday support/resistance)
    h3 = pivot_point + (range_1d * 1.1 / 4)
    l3 = pivot_point - (range_1d * 1.1 / 4)
    
    # Align to 6h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high_20_aligned[i]) or np.isnan(lowest_low_20_aligned[i]) or
            np.isnan(weekly_midpoint_aligned[i]) or np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below weekly midpoint OR below L3 (intraday support fails)
            if close[i] < weekly_midpoint_aligned[i] or close[i] < l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above weekly midpoint OR above H3 (intraday resistance fails)
            if close[i] > weekly_midpoint_aligned[i] or close[i] > h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need volume confirmation
            volume_confirmed = volume[i] > 1.5 * volume_ma[i]
            
            if volume_confirmed:
                # Long: price breaks above weekly Donchian high WITH weekly bullish bias
                if close[i] > highest_high_20_aligned[i] and close[i] > weekly_midpoint_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below weekly Donchian low WITH weekly bearish bias
                elif close[i] < lowest_low_20_aligned[i] and close[i] < weekly_midpoint_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals