#!/usr/bin/env python3
"""
4h_12h_Donchian_Breakout_1dVMA_Filter_V1
Hypothesis: Use 12h Donchian breakout with 1d volume moving average filter for directional bias.
Long when price breaks above 12h Donchian upper channel with volume > 1.5x 20-day average during active session (08-20 UTC).
Short when price breaks below 12h Donchian lower channel with volume > 1.5x 20-day average during active session.
Fixed position size 0.25. Uses 1d volume filter to ensure institutional participation and reduce false breakouts.
Target: 20-40 trades/year per symbol (80-160 total over 4 years) to minimize fee drag.
Works in bull/bear via volume filter and session timing.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # 12h Donchian channels (20-period)
    donch_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_ma_20d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all data to 4h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_12h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_12h, donch_low)
    vol_ma_20d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20d)
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # need enough for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(vol_ma_20d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-day average
        vol_filter = volume[i] > 1.5 * vol_ma_20d_aligned[i]
        
        # Only trade during active session
        in_session = session_mask[i]
        
        if position == 0:
            # Long: price breaks above 12h Donchian upper channel with volume filter during session
            if close[i] > donch_high_aligned[i] and vol_filter and in_session:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12h Donchian lower channel with volume filter during session
            elif close[i] < donch_low_aligned[i] and vol_filter and in_session:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns below 12h Donchian upper channel or outside session
            if close[i] < donch_high_aligned[i] or not in_session:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above 12h Donchian lower channel or outside session
            if close[i] > donch_low_aligned[i] or not in_session:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_12h_Donchian_Breakout_1dVMA_Filter_V1"
timeframe = "4h"
leverage = 1.0