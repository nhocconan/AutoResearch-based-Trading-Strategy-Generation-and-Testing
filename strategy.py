#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1-week Donchian channel breakout + volume filter.
# Uses weekly Donchian breakouts (20-period) with volume > 1.5x average to capture strong trends.
# Designed for low trade frequency (target: 15-30 trades/year) to minimize fee drift.
# Works in bull/bear markets by capturing breakouts with institutional volume confirmation.

name = "12h_1w_donchian_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Upper band: highest high of last 20 weeks
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low of last 20 weeks
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Calculate 1w volume average (20-period)
    volume_1w = df_1w['volume'].values
    vol_avg_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_avg_aligned = align_htf_to_ltf(prices, df_1w, vol_avg_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from index 20 to ensure Donchian and volume averages are valid
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_avg_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: current volume > 1.5 * 1w average volume
        vol_filter = volume[i] > 1.5 * vol_avg_aligned[i]
        
        # Entry conditions: price breaks through Donchian levels with volume filter
        long_entry = (high[i] > donchian_high_aligned[i] and vol_filter)
        short_entry = (low[i] < donchian_low_aligned[i] and vol_filter)
        
        # Exit conditions: price returns to opposite Donchian level
        if position == 1 and low[i] < donchian_low_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and high[i] > donchian_high_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals