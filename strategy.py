#!/usr/bin/env python3
"""
4h Donchian Breakout with 1d Trend and Volume Confirmation
Hypothesis: Donchian(20) breakouts capture momentum. Combined with 1d EMA trend filter and volume >1.5x average to avoid false signals. Works in bull/bear by aligning with higher timeframe trend. Targets 20-40 trades/year on 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_1d_trend_volume_v2"
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
    
    # 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter (>1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(high_max[i]) or 
            np.isnan(low_min[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below lower Donchian OR trend turns bearish
            if (close[i] <= low_min[i] or 
                close[i] <= ema_50_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price breaks above upper Donchian OR trend turns bullish
            if (close[i] >= high_max[i] or 
                close[i] >= ema_50_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat, look for entry
            # Long: price breaks above upper Donchian, uptrend, volume
            if (close[i] > high_max[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.30
            # Short: price breaks below lower Donchian, downtrend, volume
            elif (close[i] < low_min[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.30
    
    return signals