#!/usr/bin/env python3
# 6h_weekly_donchian_pivot_volume_v1
# Hypothesis: 6h strategy using weekly Donchian breakout with 1d pivot direction and volume confirmation.
# In trending markets, price breaks weekly Donchian channels; in ranging markets (2025+), 
# price respects 1d pivot levels. Volume confirms breakout validity. Uses discrete sizing (0.0, ±0.25) 
# to minimize fee churn. Target: 50-150 total trades over 4 years.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_donchian_pivot_volume_v1"
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
    
    # 1w HTF data for Donchian channels (20-period)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Donchian channels (20-period)
    high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian to 6h timeframe
    donchian_high = align_htf_to_ltf(prices, df_1w, high_20)
    donchian_low = align_htf_to_ltf(prices, df_1w, low_20)
    
    # 1d HTF data for pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous 1d pivot levels
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla-like levels from 1d
    r3 = close_1d + (range_1d * 1.1 / 4)
    s3 = close_1d - (range_1d * 1.1 / 4)
    r4 = close_1d + (range_1d * 1.1 / 2)
    s4 = close_1d - (range_1d * 1.1 / 2)
    
    # Align 1d pivot levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: price moves below weekly Donchian low or 1d S3
            if close[i] < donchian_low[i] or close[i] < s3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price moves above weekly Donchian high or 1d R3
            if close[i] > donchian_high[i] or close[i] > r3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed:
                # Long entry: price breaks above weekly Donchian high with volume
                if close[i] > donchian_high[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price breaks below weekly Donchian low with volume
                elif close[i] < donchian_low[i]:
                    position = -1
                    signals[i] = -0.25
                # Long entry: price touches 1d S3 with volume (mean reversion in ranging)
                elif close[i] <= s3_aligned[i] and low[i] <= s3_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price touches 1d R3 with volume (mean reversion in ranging)
                elif close[i] >= r3_aligned[i] and high[i] >= r3_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals