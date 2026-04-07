#!/usr/bin/env python3
"""
4h Donchian Breakout + 12h EMA Trend + Volume Confirmation
Hypothesis: 4-hour Donchian channel breakouts (20-period) capture significant price moves.
Trend filtered by 12-hour EMA(21) ensures directional alignment with higher timeframe.
Volume > 1.5x average confirms institutional participation. Designed for 4h timeframe 
to balance trade frequency and signal quality in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_12h_trend_volume_v1"
timeframe = "4h"
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
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # 12h EMA(21) for trend filter
    ema_21 = df_12h['close'].ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h = align_htf_to_ltf(prices, df_12h, ema_21)
    
    # 4h Donchian channel (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter (>1.5x 20-period average on 4h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(ema_21_4h[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower band or trend reverses
            if close[i] < low_min[i] or close[i] < ema_21_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper band or trend reverses
            if close[i] > high_max[i] or close[i] > ema_21_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout long with trend alignment
            if (close[i] > high_max[i] and 
                close[i] > ema_21_4h[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Breakout short with trend alignment
            elif (close[i] < low_min[i] and 
                  close[i] < ema_21_4h[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals