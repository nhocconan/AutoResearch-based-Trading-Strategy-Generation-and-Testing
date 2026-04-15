# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
12h_Donchian_20_Volume_Confirmation
Hypothesis: On 12h timeframe, price breaking above/below 20-bar Donchian channel with volume > 1.5x 20-bar median volume
captures breakout moves. Works in bull (upside breakouts) and bear (downside breakouts). 
Uses 1d timeframe for trend filter: only take longs when price > 1d EMA50, shorts when price < 1d EMA50.
Designed to generate ~15-30 trades/year per symbol, staying within limits.
"""

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
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 20-bar Donchian channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current > 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Long: price breaks above Donchian high, volume confirmation, price above 1d EMA50
        if (close[i] > highest_high[i] and 
            volume[i] > vol_threshold[i] and 
            close[i] > ema_1d_aligned[i]):
            signals[i] = 0.25
        
        # Short: price breaks below Donchian low, volume confirmation, price below 1d EMA50
        elif (close[i] < lowest_low[i] and 
              volume[i] > vol_threshold[i] and 
              close[i] < ema_1d_aligned[i]):
            signals[i] = -0.25
        
        # Exit: price returns to middle of Donchian channel
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] <= (highest_high[i] + lowest_low[i]) / 2) or
               (signals[i-1] == -0.25 and close[i] >= (highest_high[i] + lowest_low[i]) / 2))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "12h_Donchian_20_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0