#!/usr/bin/env python3
"""
6h Donchian Breakout + Weekly Trend + Volume Confirmation v1
Hypothesis: Donchian(20) breakouts on 6h timeframe, filtered by weekly trend direction
(1w EMA) and volume spikes, capture strong momentum moves while avoiding false breakouts.
Weekly trend filter ensures we only trade in the direction of the dominant trend,
improving performance in both bull and bear markets. Target: 12-37 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_breakout_weekly_trend_volume_v1"
timeframe = "6h"
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
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA(21) for trend filter
    ema_21w = df_1w['close'].ewm(span=21, adjust=False, min_periods=21).mean()
    ema_21w_6h = align_htf_to_ltf(prices, df_1w, ema_21w.values)
    
    # Donchian channel (20-period) on 6h
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter (>1.5x 20-period average on 6h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_21w_6h[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower band or trend reverses
            if close[i] <= low_20[i] or close[i] < ema_21w_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper band or trend reverses
            if close[i] >= high_20[i] or close[i] > ema_21w_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long breakout with trend alignment
            if (close[i] >= high_20[i] and 
                close[i] > ema_21w_6h[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short breakout with trend alignment
            elif (close[i] <= low_20[i] and 
                  close[i] < ema_21w_6h[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals