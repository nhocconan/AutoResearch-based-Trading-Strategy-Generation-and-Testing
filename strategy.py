#!/usr/bin/env python3
"""
4h_1d_donchian_breakout_volume
Uses Donchian channel breakout on 4h with 1d volume confirmation.
Long when price breaks above 20-period upper band and 1d volume > 1.5x average.
Short when price breaks below 20-period lower band and 1d volume > 1.5x average.
Exit when price returns to the middle of the Donchian channel.
Designed for low trade frequency (target: 20-40 trades/year) to minimize fee drag.
Works in trending markets by capturing breakouts and avoiding whipsaws via volume filter.
"""

name = "4h_1d_donchian_breakout_volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel on 4h: 20-period high/low
    lookback = 20
    highest = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    middle = (highest + lowest) / 2
    
    # Get daily volume for confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    # Daily volume moving average: 20-period
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    # Align to 4h timeframe
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):
        # Skip if data not ready
        if (np.isnan(highest[i]) or np.isnan(lowest[i]) or 
            np.isnan(middle[i]) or np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: price breaks above upper band with high volume
        if (close[i] > highest[i] and volume[i] > vol_ma_1d_aligned[i] * 1.5 and 
            position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: price breaks below lower band with high volume
        elif (close[i] < lowest[i] and volume[i] > vol_ma_1d_aligned[i] * 1.5 and 
              position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: price returns to middle of channel
        elif position == 1 and close[i] < middle[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > middle[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals