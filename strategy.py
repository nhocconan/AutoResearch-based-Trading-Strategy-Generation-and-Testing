#!/usr/bin/env python3
"""
#100982 - 12h_Donchian20_1dTrend_Volume
Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
Works in bull markets (breakouts with trend) and bear markets (false breakouts reversed via volatility filter).
Target: 15-25 trades/year to minimize fee drag. Uses discrete position sizing (0.30).
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 12h Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 1.5x 50-period average
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(high_roll[i]) or 
            np.isnan(low_roll[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: price breaks above Donchian high, above 1d EMA34, volume spike
        if (close[i] > high_roll[i] and 
            close[i] > ema34_1d_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.30
            position = 1
        # Short condition: price breaks below Donchian low, below 1d EMA34, volume spike
        elif (close[i] < low_roll[i] and 
              close[i] < ema34_1d_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.30
            position = -1
        # Exit conditions: price returns to Donchian midpoint (mean reversion)
        elif position == 1 and close[i] < (high_roll[i] + low_roll[i]) / 2:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > (high_roll[i] + low_roll[i]) / 2:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.30
            elif position == -1:
                signals[i] = -0.30
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_Donchian20_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0