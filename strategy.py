#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with daily Donchian breakout + volume confirmation.
# Uses daily Donchian channels (20-day high/low) to capture momentum after volatility expansion.
# Long when price breaks above previous day's 20-day high with volume > 1.5x average,
# short when breaks below previous day's 20-day low with volume > 1.5x average.
# Exit when price returns to the opposite Donchian band (mean reversion within channel).
# Designed for low trade frequency (~20-40/year) to minimize fee decay while capturing volatility breakouts.
# Works in bull/bear markets by trading volatility expansions in either direction.

name = "12h_1d_donchian_breakout_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 25:
        return np.zeros(n)
    
    # Calculate daily Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 20-day high and low
    high_20 = np.full_like(high_1d, np.nan)
    low_20 = np.full_like(low_1d, np.nan)
    
    for i in range(19, len(high_1d)):
        high_20[i] = np.max(high_1d[i-19:i+1])
        low_20[i] = np.min(low_1d[i-19:i+1])
    
    # Calculate daily average volume (20-period)
    volume_1d = df_1d['volume'].values
    vol_avg_20 = np.full_like(volume_1d, np.nan, dtype=float)
    
    for i in range(19, len(volume_1d)):
        vol_avg_20[i] = np.mean(volume_1d[i-19:i+1])
    
    # Align daily levels to 12h timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from index 1 to ensure we have previous day's data
    for i in range(1, n):
        # Skip if any required data is invalid
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or
            np.isnan(vol_avg_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: current volume > 1.5 * daily average volume
        vol_filter = volume[i] > 1.5 * vol_avg_aligned[i]
        
        # Entry conditions: price breaks above/below Donchian bands with volume
        long_break = high[i] > high_20_aligned[i] and vol_filter
        short_break = low[i] < low_20_aligned[i] and vol_filter
        
        # Exit conditions: price returns to opposite Donchian band (mean reversion)
        exit_long = low[i] <= low_20_aligned[i] if not np.isnan(low_20_aligned[i]) else False
        exit_short = high[i] >= high_20_aligned[i] if not np.isnan(high_20_aligned[i]) else False
        
        if long_break and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_break and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals