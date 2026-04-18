#!/usr/bin/env python3
"""
1d DailyDonchianBreakout_WeeklyTrendFilter
Based on Donchian channel breakout with weekly trend filter.
Long when price breaks above weekly Donchian high with volume confirmation.
Short when price breaks below weekly Donchian low with volume confirmation.
Designed for low trade frequency with clear trend-following edge.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channel (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Volume spike detection (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above weekly Donchian high with volume spike
            if price > donchian_high_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian low with volume spike
            elif price < donchian_low_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position management
            signals[i] = 0.25
            # Exit: price breaks below weekly Donchian low
            if price < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.25
            # Exit: price breaks above weekly Donchian high
            if price > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_DailyDonchianBreakout_WeeklyTrendFilter"
timeframe = "1d"
leverage = 1.0