#!/usr/bin/env python3
"""
1d_Donchian_20_Breakout_1wTrend_Volume
Hypothesis: Trade daily breakouts of Donchian(20) channels only when aligned with weekly trend (EMA200) and confirmed by volume spike (>2x average). Uses weekly timeframe for trend direction and daily for signal execution, targeting 15-25 trades/year with low fee impact. Works in both bull and bear markets by requiring trend alignment and volume confirmation.
"""

name = "1d_Donchian_20_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period) on daily data
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Get weekly trend filter (EMA200)
    weekly_close = df_1w['close'].values
    ema_200_1w = pd.Series(weekly_close).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Get daily volume for confirmation
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma20, out=np.zeros_like(volume), where=vol_ma20!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or 
            np.isnan(ema_200_1w_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend using aligned close
        weekly_close_aligned = align_htf_to_ltf(prices, df_1w, weekly_close)
        if np.isnan(weekly_close_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        trend_up = weekly_close_aligned[i] > ema_200_1w_aligned[i]
        trend_down = weekly_close_aligned[i] < ema_200_1w_aligned[i]
        
        if position == 0:
            # Long breakout: price breaks above Donchian high with upward trend and volume spike
            if (close[i] > high_20[i] and 
                trend_up and 
                vol_ratio[i] > 2.0):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below Donchian low with downward trend and volume spike
            elif (close[i] < low_20[i] and 
                  trend_down and 
                  vol_ratio[i] > 2.0):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian low or trend turns down
            if close[i] < low_20[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian high or trend turns up
            if close[i] > high_20[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals