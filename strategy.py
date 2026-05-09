#\n#!/usr/bin/env python3
# Hypothesis: 12-hour timeframe strategy using 1-day Donchian breakout with 12-hour EMA trend filter and volume confirmation
# Long when price breaks above 1-day Donchian Upper Channel with rising 12h EMA25 and volume > 1.5x average
# Short when price breaks below 1-day Donchian Lower Channel with falling 12h EMA25 and volume > 1.5x average
# Exit when price crosses the 12h EMA25 in the opposite direction
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25
# Designed to capture multi-day trends while avoiding whipsaws in ranging markets

name = "12h_Donchian_Breakout_12hEMA25_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1-day Donchian channels (20-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Previous day's Donchian channels (using shifted data to avoid look-ahead)
    prev_high = df_1d['high'].shift(1)
    prev_low = df_1d['low'].shift(1)
    
    # Calculate 20-period rolling max/min
    donchian_upper = prev_high.rolling(window=20, min_periods=20).max().values
    donchian_lower = prev_low.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 12h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    
    # Calculate 12-hour EMA25 for trend filter
    ema25 = pd.Series(close).ewm(span=25, adjust=False, min_periods=25).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(ema25[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Donchian upper, EMA25 rising, volume spike
            if (close[i] > donchian_upper_aligned[i] and 
                ema25[i] > ema25[i-1] and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian lower, EMA25 falling, volume spike
            elif (close[i] < donchian_lower_aligned[i] and 
                  ema25[i] < ema25[i-1] and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below EMA25
            if close[i] < ema25[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above EMA25
            if close[i] > ema25[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals