#!/usr/bin/env python3
"""
1h Donchian Breakout + 4h Trend + Volume Filter
Hypothesis: 1-hour Donchian channel breakouts with 4-hour EMA trend alignment and volume confirmation capture momentum moves while reducing false breakouts. The 4h EMA filter ensures trades align with higher-timeframe trend, improving win rate in both bull and bear markets. Target: 15-30 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_donchian_breakout_4h_trend_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    ema_4h = df_4h['close'].ewm(span=20, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Volume filter: 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    # 1h Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(vol_filter[i]) or 
            np.isnan(high_max[i]) or np.isnan(low_min[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low or trend reverses
            if close[i] <= low_min[i] or close[i] < ema_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high or trend reverses
            if close[i] >= high_max[i] or close[i] > ema_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Breakout long at Donchian high with trend alignment
            if (close[i] >= high_max[i] and 
                close[i] > ema_4h_aligned[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.20
            # Breakout short at Donchian low with trend alignment
            elif (close[i] <= low_min[i] and 
                  close[i] < ema_4h_aligned[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.20
    
    return signals