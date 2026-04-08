#!/usr/bin/env python3
# 4h_12h_donchian_volume_v1
# Hypothesis: 4h Donchian breakout with 12h trend filter and volume confirmation.
# Long when: price breaks above 4h Donchian upper channel (20), 12h EMA25 rising, volume > 1.5x 20-period average.
# Short when: price breaks below 4h Donchian lower channel (20), 12h EMA25 falling, volume > 1.5x 20-period average.
# Exit when price crosses 4h EMA10 in opposite direction.
# Target: 20-40 trades/year with strict entry conditions to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_donchian_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h EMA10 for exit
    ema_10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # 12h EMA25 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h_25 = pd.Series(close_12h).ewm(span=25, adjust=False, min_periods=25).mean().values
    ema_12h_25_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_25)
    
    # Volume filter: 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 40  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_10[i]) or np.isnan(ema_12h_25_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.5 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: Price below EMA10
            if close[i] < ema_10[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price above EMA10
            if close[i] > ema_10[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price above Donchian upper, 12h EMA25 rising, volume surge
            if (close[i] > high_20[i] and 
                ema_12h_25_aligned[i] > ema_12h_25_aligned[i-1] and 
                vol_surge):
                position = 1
                signals[i] = 0.25
            # Short entry: Price below Donchian lower, 12h EMA25 falling, volume surge
            elif (close[i] < low_20[i] and 
                  ema_12h_25_aligned[i] < ema_12h_25_aligned[i-1] and 
                  vol_surge):
                position = -1
                signals[i] = -0.25
    
    return signals