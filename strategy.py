#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d timeframe with weekly Donchian breakout + volume confirmation.
# Uses weekly price channels to capture long-term trend momentum.
# Long when price breaks above weekly high with volume > 1.5x average,
# short when breaks below weekly low with volume > 1.5x average.
# Exit when price returns to weekly midpoint (mean reversion).
# Designed for low trade frequency (~10-25/year) to minimize fee decay.
# Works in bull/bear markets by trading breakouts in either direction.

name = "1d_1w_donchian_breakout_volume_v1"
timeframe = "1d"
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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate rolling max/min for Donchian channels
    def rolling_max(arr, window):
        res = np.full_like(arr, np.nan, dtype=float)
        for i in range(window-1, len(arr)):
            res[i] = np.max(arr[i-window+1:i+1])
        return res
    
    def rolling_min(arr, window):
        res = np.full_like(arr, np.nan, dtype=float)
        for i in range(window-1, len(arr)):
            res[i] = np.min(arr[i-window+1:i+1])
        return res
    
    # Weekly Donchian channels
    upper_donch = rolling_max(high_1w, 20)
    lower_donch = rolling_min(low_1w, 20)
    mid_donch = (upper_donch + lower_donch) / 2
    
    # Calculate weekly average volume (20-period)
    volume_1w = df_1w['volume'].values
    vol_avg_20 = np.full_like(volume_1w, np.nan, dtype=float)
    for i in range(19, len(volume_1w)):
        vol_avg_20[i] = np.mean(volume_1w[i-19:i+1])
    
    # Align weekly levels to daily timeframe
    upper_donch_aligned = align_htf_to_ltf(prices, df_1w, upper_donch)
    lower_donch_aligned = align_htf_to_ltf(prices, df_1w, lower_donch)
    mid_donch_aligned = align_htf_to_ltf(prices, df_1w, mid_donch)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1w, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(n):
        # Skip if any required data is invalid
        if (np.isnan(upper_donch_aligned[i]) or np.isnan(lower_donch_aligned[i]) or
            np.isnan(mid_donch_aligned[i]) or np.isnan(vol_avg_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: current volume > 1.5 * weekly average volume
        vol_filter = volume[i] > 1.5 * vol_avg_aligned[i]
        
        # Entry conditions: price breaks above/below weekly Donchian with volume
        long_break = high[i] > upper_donch_aligned[i] and vol_filter
        short_break = low[i] < lower_donch_aligned[i] and vol_filter
        
        # Exit conditions: price returns to weekly midpoint (mean reversion)
        exit_long = low[i] <= mid_donch_aligned[i]
        exit_short = high[i] >= mid_donch_aligned[i]
        
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