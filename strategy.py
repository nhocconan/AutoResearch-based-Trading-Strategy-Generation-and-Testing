#!/usr/bin/env python3
# 12h_1d_breakout_volume_v1
# Hypothesis: Trade breakouts of 1d high/low channels with volume confirmation on 12h timeframe.
# Uses 20-period high/low channels from daily timeframe for structure, volume surge for confirmation.
# Works in bull markets (breakouts above channel) and bear markets (breakdowns below channel).
# Target: 15-35 trades/year on 12h timeframe with strict entry conditions to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_breakout_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily high/low channels for structure
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 20-period highest high and lowest low from daily
    high_channel = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_channel = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align daily channels to 12h timeframe
    high_channel_aligned = align_htf_to_ltf(prices, df_1d, high_channel)
    low_channel_aligned = align_htf_to_ltf(prices, df_1d, low_channel)
    
    # Volume confirmation: 12h volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 20  # Ensure channel calculation is ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_channel_aligned[i]) or np.isnan(low_channel_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 2.0 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: price breaks below lower channel OR volatility stop
            if close[i] < low_channel_aligned[i] or close[i] < high[i] - 3.0 * (high[i] - low[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above upper channel OR volatility stop
            if close[i] > high_channel_aligned[i] or close[i] > low[i] + 3.0 * (high[i] - low[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above upper channel with volume surge
            if close[i] > high_channel_aligned[i] and vol_surge:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below lower channel with volume surge
            elif close[i] < low_channel_aligned[i] and vol_surge:
                position = -1
                signals[i] = -0.25
    
    return signals