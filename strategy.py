#!/usr/bin/env python3
# 6h_weekly_donchian_breakout_volume_v3
# Hypothesis: 6h strategy using weekly Donchian channel breakouts with volume confirmation.
# Long: Price breaks above weekly Donchian(20) high with volume > 1.5x 20-period average
# Short: Price breaks below weekly Donchian(20) low with volume > 1.5x 20-period average
# Exit: Price returns to weekly Donchian(10) midpoint or opposite 20-period break
# Uses 6h primary timeframe with 1w HTF for Donchian channels.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Works in both bull and bear markets by capturing breakouts with volume confirmation.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_donchian_breakout_volume_v3"
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
    
    # Calculate volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly Donchian channels
    # Donchian(20) for breakout signals
    donch_high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    # Donchian(10) for exit signals (midpoint)
    donch_high_10 = pd.Series(high_1w).rolling(window=10, min_periods=10).max().values
    donch_low_10 = pd.Series(low_1w).rolling(window=10, min_periods=10).min().values
    donch_mid_10 = (donch_high_10 + donch_low_10) / 2.0
    
    # Align weekly Donchian levels to 6h timeframe
    donch_high_20_aligned = align_htf_to_ltf(prices, df_1w, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_1w, donch_low_20)
    donch_mid_10_aligned = align_htf_to_ltf(prices, df_1w, donch_mid_10)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(40, n):  # Start after warmup period for all indicators
        # Skip if any required data is NaN
        if (np.isnan(donch_high_20_aligned[i]) or np.isnan(donch_low_20_aligned[i]) or 
            np.isnan(donch_mid_10_aligned[i]) or np.isnan(volume_ma[i]) or
            np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: Price returns to weekly Donchian(10) midpoint or breaks below weekly Donchian(20) low
            if close[i] <= donch_mid_10_aligned[i] or close[i] < donch_low_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price returns to weekly Donchian(10) midpoint or breaks above weekly Donchian(20) high
            if close[i] >= donch_mid_10_aligned[i] or close[i] > donch_high_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price breaks above weekly Donchian(20) high with volume confirmation
            if close[i] > donch_high_20_aligned[i] and volume_confirmed:
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below weekly Donchian(20) low with volume confirmation
            elif close[i] < donch_low_20_aligned[i] and volume_confirmed:
                position = -1
                signals[i] = -0.25
    
    return signals