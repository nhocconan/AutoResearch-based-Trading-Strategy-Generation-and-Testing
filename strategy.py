#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d Bollinger Band squeeze + volume confirmation
# Donchian breakouts capture momentum; 1d BB squeeze identifies low volatility primed for expansion
# Volume confirmation ensures breakout authenticity with conviction
# Works in bull/bear: BB squeeze precedes moves in both directions, Donchian captures the breakout
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25-0.30

name = "12h_1d_bb_squeeze_donchian_breakout_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Bollinger Bands (20, 2)
    close_1d = df_1d['close'].values
    sma_20 = np.full(len(close_1d), np.nan)
    std_20 = np.full(len(close_1d), np.nan)
    
    for i in range(len(close_1d)):
        if i < 19:
            sma_20[i] = np.nan
            std_20[i] = np.nan
        else:
            sma_20[i] = np.mean(close_1d[i-19:i+1])
            std_20[i] = np.std(close_1d[i-19:i+1])
    
    upper_band = sma_20 + 2 * std_20
    lower_band = sma_20 - 2 * std_20
    bb_width = (upper_band - lower_band) / sma_20
    
    # BB squeeze: width < 20-period average width (low volatility)
    avg_bb_width = np.full(len(bb_width), np.nan)
    for i in range(len(bb_width)):
        if i < 19:
            avg_bb_width[i] = np.nan
        else:
            avg_bb_width[i] = np.mean(bb_width[i-19:i+1])
    
    squeeze = bb_width < 0.8 * avg_bb_width  # Squeeze when width is 20% below average
    
    # Align BB squeeze data to 12h timeframe (wait for daily close)
    squeeze_aligned = align_htf_to_ltf(prices, df_1d, squeeze)
    
    # Calculate 12h Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(n):
        if i < 20:
            donchian_high[i] = np.nan
            donchian_low[i] = np.nan
        else:
            donchian_high[i] = np.max(high[i-20:i])
            donchian_low[i] = np.min(low[i-20:i])
    
    # Calculate 20-period average volume for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(squeeze_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price < Donchian low OR BB squeeze ends (volatility expansion)
            if close[i] < donchian_low[i] or not squeeze_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > Donchian high OR BB squeeze ends (volatility expansion)
            if close[i] > donchian_high[i] or not squeeze_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation, BB squeeze, and Donchian breakout
            if volume_confirmed and squeeze_aligned[i]:
                # Long entry: price > Donchian high
                if close[i] > donchian_high[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price < Donchian low
                elif close[i] < donchian_low[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals