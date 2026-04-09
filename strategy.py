#!/usr/bin/env python3
# 6h_weekly_pivot_donchian_volume_v1
# Hypothesis: 6h strategy using weekly pivot points for direction and Donchian(20) breakouts for entry, with volume confirmation.
# Long: Price breaks above Donchian(20) high with volume > 1.5x 20-period average AND weekly pivot bias bullish (price > weekly PP).
# Short: Price breaks below Donchian(20) low with volume > 1.5x 20-period average AND weekly pivot bias bearish (price < weekly PP).
# Exit: Price returns to Donchian(20) midpoint.
# Uses weekly pivot points from 1w timeframe as bias filter.
# Volume confirmation filters breakouts. Target: 12-37 trades/year (50-150 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_donchian_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Get 1w data for weekly pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot point (PP) = (High + Low + Close) / 3
    weekly_pp = (high_1w + low_1w + close_1w) / 3.0
    
    # Align weekly PP to 6h
    weekly_pp_aligned = align_htf_to_ltf(prices, df_1w, weekly_pp)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i]) or
            np.isnan(weekly_pp_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: Price returns to Donchian midpoint
            if close[i] <= donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price returns to Donchian midpoint
            if close[i] >= donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price breaks above Donchian high with volume confirmation AND bullish weekly bias (price > weekly PP)
            if (close[i] > donchian_high[i] and volume_confirmed and close[i] > weekly_pp_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian low with volume confirmation AND bearish weekly bias (price < weekly PP)
            elif (close[i] < donchian_low[i] and volume_confirmed and close[i] < weekly_pp_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals