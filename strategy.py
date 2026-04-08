#!/usr/bin/env python3
# 1d_1w_donchian_breakout_volume_v1
# Hypothesis: Daily Donchian(20) breakout with weekly trend filter (EMA200) and volume confirmation.
# Long when price breaks above upper Donchian channel, weekly EMA200 rising, and volume > 1.5x 20-day average.
# Short when price breaks below lower Donchian channel, weekly EMA200 falling, and volume > 1.5x 20-day average.
# Exit when price crosses the opposite Donchian channel boundary.
# Target: 15-25 trades/year (~60-100 total over 4 years) with strict entry conditions to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_donchian_breakout_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_1w_200 = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_1w_200_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_200)
    
    # Daily Donchian(20) channels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 200  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_1w_200_aligned[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.5 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: Price below lower Donchian channel
            if close[i] < low_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price above upper Donchian channel
            if close[i] > high_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price above upper Donchian channel, weekly EMA200 rising, volume surge
            if (close[i] > high_20[i] and 
                ema_1w_200_aligned[i] > ema_1w_200_aligned[i-1] and 
                vol_surge):
                position = 1
                signals[i] = 0.25
            # Short entry: Price below lower Donchian channel, weekly EMA200 falling, volume surge
            elif (close[i] < low_20[i] and 
                  ema_1w_200_aligned[i] < ema_1w_200_aligned[i-1] and 
                  vol_surge):
                position = -1
                signals[i] = -0.25
    
    return signals