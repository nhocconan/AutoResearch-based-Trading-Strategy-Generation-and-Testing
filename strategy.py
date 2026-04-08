#!/usr/bin/env python3
"""
1d Weekly Donchian Breakout with Volume Confirmation
Hypothesis: Weekly Donchian breakouts capture long-term trends. Volume confirmation filters false breakouts. Works in both bull and bear markets by trading breakouts in either direction. Targets 8-25 trades/year on 1d timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_donchian_breakout_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly Donchian channels (20 periods)
    df_1w = get_htf_data(prices, '1w')
    period20_high = df_1w['high'].rolling(window=20, min_periods=20).max().values
    period20_low = df_1w['low'].rolling(window=20, min_periods=20).min().values
    donchian_high = period20_high
    donchian_low = period20_low
    
    # Align Donchian channels to daily timeframe (shifted by 1 for completed weekly bars)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Volume filter (>1.5x 20-day average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below weekly Donchian low
            if close[i] <= donchian_low_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above weekly Donchian high
            if close[i] >= donchian_high_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price breaks above weekly Donchian high with volume
            if close[i] > donchian_high_aligned[i] and vol_filter[i]:
                position = 1
                signals[i] = 0.25
            # Short: price breaks below weekly Donchian low with volume
            elif close[i] < donchian_low_aligned[i] and vol_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals