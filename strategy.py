#!/usr/bin/env python3
# 1d_1w_donchian_breakout_volume
# Hypothesis: Daily Donchian(20) breakouts with weekly trend filter and volume confirmation.
# Long when price breaks above 20-day high with weekly uptrend (price > weekly SMA50) and volume surge.
# Short when price breaks below 20-day low with weekly downtrend (price < weekly SMA50) and volume surge.
# Uses weekly trend to avoid counter-trend trades in strong trends, reducing whipsaw.
# Target: 15-25 trades/year on 1d timeframe with strict entry conditions to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_donchian_breakout_volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly trend filter: SMA50 on weekly close
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    sma50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    sma50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma50_1w)
    
    # Daily Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.8x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 100  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(sma50_1w_aligned[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.8 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: price breaks below 20-day low OR weekly trend turns down
            if low[i] <= low_20[i] or close[i] < sma50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above 20-day high OR weekly trend turns up
            if high[i] >= high_20[i] or close[i] > sma50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above 20-day high with weekly uptrend and volume surge
            if high[i] > high_20[i] and close[i] > sma50_1w_aligned[i] and vol_surge:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below 20-day low with weekly downtrend and volume surge
            elif low[i] < low_20[i] and close[i] < sma50_1w_aligned[i] and vol_surge:
                position = -1
                signals[i] = -0.25
    
    return signals