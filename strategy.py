#!/usr/bin/env python3
"""
6h Donchian Breakout with 1d Volume Confirmation and Trend Filter
Hypothesis: Breakouts from Donchian channels (20-period high/low) capture strong momentum.
Filtered by 1d EMA trend to avoid counter-trend trades and volume surge to confirm institutional interest.
Works in both bull and bear markets by aligning with higher timeframe trend.
Target: 20-40 trades per year on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_breakout_1d_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
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
    
    # 1d volume average for confirmation
    vol_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_20_1d)
    
    # Donchian channels (20-period) on 6s timeframe
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_20_1d_aligned[i]) or 
            np.isnan(high_20[i]) or np.isnan(low_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below lower Donchian OR trend turns bearish
            if (close[i] <= low_20[i] or 
                close[i] <= ema_50_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above upper Donchian OR trend turns bullish
            if (close[i] >= high_20[i] or 
                close[i] >= ema_50_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price breaks above upper Donchian, uptrend, volume surge
            if (close[i] > high_20[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume[i] > vol_20_1d_aligned[i] * 1.5):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below lower Donchian, downtrend, volume surge
            elif (close[i] < low_20[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume[i] > vol_20_1d_aligned[i] * 1.5):
                position = -1
                signals[i] = -0.25
    
    return signals