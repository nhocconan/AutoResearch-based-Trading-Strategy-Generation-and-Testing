#!/usr/bin/env python3
# 1d_weekly_donchian_breakout_volume_v1
# Hypothesis: Weekly Donchian breakout with volume confirmation on daily chart.
# Enter long when price breaks above 20-week Donchian high, volume > 1.5x average volume.
# Enter short when price breaks below 20-week Donchian low, volume > 1.5x average volume.
# Exit when price returns to the weekly Donchian midpoint.
# Designed for 10-30 trades/year on 1d to avoid fee drag. Works in bull/bear via breakout logic.

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
    
    # Weekly Donchian channel (20-period)
    lookback = 20
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    donchian_high_1w = np.full(len(df_1w), np.nan)
    donchian_low_1w = np.full(len(df_1w), np.nan)
    
    for i in range(lookback, len(df_1w)):
        donchian_high_1w[i] = np.max(high_1w[i-lookback:i])
        donchian_low_1w[i] = np.min(low_1w[i-lookback:i])
    
    # Align weekly Donchian levels to daily
    donchian_high_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_1w)
    donchian_low_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_1w)
    
    # Volume average (20-period) for confirmation
    vol_avg = np.full(n, np.nan)
    for i in range(20, n):
        vol_avg[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(50, 20)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high_1w_aligned[i]) or np.isnan(donchian_low_1w_aligned[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirmed = volume[i] > 1.5 * vol_avg[i]
        
        if position == 1:  # Long position
            # Exit: price returns to weekly Donchian midpoint
            midpoint = (donchian_high_1w_aligned[i] + donchian_low_1w_aligned[i]) / 2
            if close[i] < midpoint:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to weekly Donchian midpoint
            midpoint = (donchian_high_1w_aligned[i] + donchian_low_1w_aligned[i]) / 2
            if close[i] > midpoint:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: breakout above weekly Donchian high with volume confirmation
            if (close[i] > donchian_high_1w_aligned[i] and vol_confirmed):
                position = 1
                signals[i] = 0.25
            # Short entry: breakout below weekly Donchian low with volume confirmation
            elif (close[i] < donchian_low_1w_aligned[i] and vol_confirmed):
                position = -1
                signals[i] = -0.25
    
    return signals