#!/usr/bin/env python3
# 6h_weekly_donchian_breakout_volume_v1
# Hypothesis: 6h strategy using weekly Donchian channels with volume confirmation.
# In ranging markets (2025+), price tends to break weekly ranges with volume spikes.
# Weekly Donchian provides structural support/resistance, volume confirms breakout strength.
# Discrete sizing (0.0, ±0.25) minimizes fee churn. Target: 50-150 total trades over 4 years.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_donchian_breakout_volume_v1"
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
    
    # Weekly HTF data for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average
        volume_confirmed = volume[i] > 2.0 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: price moves below weekly Donchian low or volume dries up
            if close[i] < donchian_low_aligned[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price moves above weekly Donchian high or volume dries up
            if close[i] > donchian_high_aligned[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed:
                # Long entry: price breaks above weekly Donchian high with volume
                if close[i] > donchian_high_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price breaks below weekly Donchian low with volume
                elif close[i] < donchian_low_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals