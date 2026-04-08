#!/usr/bin/env python3
# 1d_weekly_donchian_breakout_volume_v2
# Hypothesis: Weekly Donchian channel breakout on daily timeframe with volume confirmation.
# Long when daily close breaks above weekly Donchian high (20-period) with volume > 1.5x average.
# Short when daily close breaks below weekly Donchian low (20-period) with volume > 1.5x average.
# Exit when price returns to weekly Donchian midpoint or volume drops below average.
# Designed for 10-25 trades/year on 1d to avoid fee drag. Works in bull/bear via breakout logic.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

name = "1d_weekly_donchian_breakout_volume_v2"
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
    
    # Weekly data for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Donchian channels (20-period)
    donchian_high = np.full_like(close_1w, np.nan)
    donchian_low = np.full_like(close_1w, np.nan)
    
    for i in range(20, len(close_1w)):
        donchian_high[i] = np.max(high_1w[i-20:i])
        donchian_low[i] = np.min(low_1w[i-20:i])
    
    # Calculate weekly Donchian midpoint
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate average volume (20-period)
    avg_volume = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Align weekly indicators to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid)
    avg_volume_aligned = align_htf_to_ltf(prices, df_1w, avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 20  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or \
           np.isnan(donchian_mid_aligned[i]) or np.isnan(avg_volume_aligned[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = volume[i] > 1.5 * avg_volume_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price returns to midpoint OR volume drops below average
            if close[i] <= donchian_mid_aligned[i] or volume[i] <= avg_volume_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to midpoint OR volume drops below average
            if close[i] >= donchian_mid_aligned[i] or volume[i] <= avg_volume_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian high with volume confirmation
            if close[i] > donchian_high_aligned[i] and volume_confirm:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian low with volume confirmation
            elif close[i] < donchian_low_aligned[i] and volume_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals