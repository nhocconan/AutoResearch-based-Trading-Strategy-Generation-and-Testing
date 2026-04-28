#!/usr/bin/env python3
"""
1h_4h1dTrend_4hVolumeBreakout
Hypothesis: Use 4h and 1d trend filters with 4h volume-confirmed breakouts on 1h timeframe.
In bull markets, trend-following breakouts capture momentum. In bear markets, trend filter reduces false signals.
Volume confirmation filters low-probability breakouts. Target: 15-30 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h and 1d data for trend filters
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate 4h Donchian breakout (20-period)
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h volume spike (>1.8x 20-period MA)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_200_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend direction from 4h EMA50 and 1d EMA200 (both must agree)
        trend_up = close[i] > ema_50_4h_aligned[i] and close[i] > ema_200_1d_aligned[i]
        trend_down = close[i] < ema_50_4h_aligned[i] and close[i] < ema_200_1d_aligned[i]
        
        # Donchian breakout conditions
        breakout_up = high[i] > high_max_20[i-1]  # Break above previous high
        breakout_down = low[i] < low_min_20[i-1]  # Break below previous low
        
        # Volume confirmation
        vol_confirm = vol_spike[i]
        
        # Entry logic: Only take breakouts in direction of both trends
        long_entry = breakout_up and trend_up and vol_confirm
        short_entry = breakout_down and trend_down and vol_confirm
        
        # Exit logic: Opposite Donchian breakout or trend disagreement
        long_exit = breakout_down or not trend_up
        short_exit = breakout_up or not trend_down
        
        if long_entry and position <= 0:
            signals[i] = 0.20
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.20
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_4h1dTrend_4hVolumeBreakout"
timeframe = "1h"
leverage = 1.0