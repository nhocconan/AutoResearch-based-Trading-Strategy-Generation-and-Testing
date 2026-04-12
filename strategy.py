#!/usr/bin/env python3
"""
6h_1d_Donchian_Breakout_v1
Hypothesis: Use 1-day Donchian channels for long-term trend direction, with 6-hour Donchian breakouts in the direction of the daily trend. Volume confirmation filters false breakouts. Works in both bull and bear markets by aligning with the higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Donchian_Breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily trend: 20-period Donchian channel
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily Donchian channels
    high_20d = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    low_20d = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    
    # Determine daily trend: 1 if above midpoint, -1 if below, 0 if inside
    mid_20d = (high_20d + low_20d) / 2
    daily_trend = np.where(close_1d := df_1d['close'].values > mid_20d, 1, 
                          np.where(close_1d < mid_20d, -1, 0))
    
    # Align daily trend to 6h timeframe
    daily_trend_aligned = align_htf_to_ltf(prices, df_1d, daily_trend)
    
    # 6-hour Donchian breakout channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean()
    vol_ratio = volume_series / vol_ma
    vol_ratio = vol_ratio.fillna(1.0).values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any data invalid
        if (np.isnan(daily_trend_aligned[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions in direction of daily trend
        long_breakout = high[i] > high_20[i] and daily_trend_aligned[i] == 1
        short_breakout = low[i] < low_20[i] and daily_trend_aligned[i] == -1
        
        # Volume confirmation
        vol_confirm = vol_ratio[i] > 1.5
        
        # Exit when price returns to the 6h Donchian midpoint
        mid_20 = (high_20[i] + low_20[i]) / 2
        long_exit = close[i] < mid_20
        short_exit = close[i] > mid_20
        
        # Signal logic
        if long_breakout and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals