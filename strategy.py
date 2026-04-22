#!/usr/bin/env python3
"""
Hypothesis: 1-day Donchian(20) breakout with 1-week EMA10 trend filter and volume confirmation.
Long when price breaks above 20-day high, weekly EMA10 rising, and volume > 1.5x 20-day average volume.
Short when price breaks below 20-day low, weekly EMA10 falling, and volume > 1.5x 20-day average volume.
Exit when price crosses back through the opposite Donchian band or weekly EMA trend reverses.
Designed for low trade frequency by requiring multiple confirmations and using daily timeframe.
Works in both bull and bear markets by following weekly trend while using daily breakouts for entries.
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
    
    # Load 1-week data for EMA10 trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema10_1w = pd.Series(close_1w).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema10_1w)
    
    # Calculate 20-day Donchian channels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-day average volume for volume confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after enough data for Donchian
        # Skip if data not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(vol_avg_20[i]) or 
            np.isnan(ema10_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above 20-day high, weekly EMA10 rising, volume confirmation
            if (close[i] > high_20[i] and 
                ema10_1w_aligned[i] > ema10_1w_aligned[i-1] and
                volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 20-day low, weekly EMA10 falling, volume confirmation
            elif (close[i] < low_20[i] and 
                  ema10_1w_aligned[i] < ema10_1w_aligned[i-1] and
                  volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price falls below 20-day low OR weekly EMA10 turns down
                if (close[i] < low_20[i] or 
                    ema10_1w_aligned[i] < ema10_1w_aligned[i-1]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price rises above 20-day high OR weekly EMA10 turns up
                if (close[i] > high_20[i] or 
                    ema10_1w_aligned[i] > ema10_1w_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Donchian20_1wEMA10_Trend_Volume"
timeframe = "1d"
leverage = 1.0