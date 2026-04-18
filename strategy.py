#!/usr/bin/env python3
"""
6h_Weekly_Donchian_Trend_Filter
Hypothesis: Uses weekly Donchian channel breakouts with 1d volume confirmation to capture strong directional moves.
Filters out false breakouts by requiring volume > 1.5x 20-period average and price to remain within weekly channel.
Designed for fewer, higher-quality trades to reduce fee drag and improve robustness in bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period)
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    
    donchian_high = np.zeros_like(high_weekly)
    donchian_low = np.zeros_like(low_weekly)
    
    for i in range(len(high_weekly)):
        if i < 20:
            donchian_high[i] = np.max(high_weekly[0:i+1]) if i >= 0 else high_weekly[i]
            donchian_low[i] = np.min(low_weekly[0:i+1]) if i >= 0 else low_weekly[i]
        else:
            donchian_high[i] = np.max(high_weekly[i-20+1:i+1])
            donchian_low[i] = np.min(low_weekly[i-20+1:i+1])
    
    # Align weekly Donchian levels to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_weekly, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_weekly, donchian_low)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    volume_ma_1d = np.zeros_like(volume_1d)
    
    for i in range(len(volume_1d)):
        if i < 20:
            volume_ma_1d[i] = np.mean(volume_1d[0:i+1]) if i >= 0 else volume_1d[i]
        else:
            volume_ma_1d[i] = np.mean(volume_1d[i-20+1:i+1])
    
    volume_spike_1d = volume_1d > (volume_ma_1d * 1.5)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = 20  # Warmup for Donchian
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Long: price breaks above weekly Donchian high with volume spike
            if close[i] > donchian_high_aligned[i] and volume_spike_aligned[i]:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: price breaks below weekly Donchian low with volume spike
            elif close[i] < donchian_low_aligned[i] and volume_spike_aligned[i]:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        
        elif position == 1:
            # Exit: price returns to weekly Donchian low or hold too long
            if close[i] < donchian_low_aligned[i] or bars_since_entry >= 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price returns to weekly Donchian high or hold too long
            if close[i] > donchian_high_aligned[i] or bars_since_entry >= 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Weekly_Donchian_Trend_Filter"
timeframe = "6h"
leverage = 1.0