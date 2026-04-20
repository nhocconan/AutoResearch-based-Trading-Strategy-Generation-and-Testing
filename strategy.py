#!/usr/bin/env python3
# 4h_12h_CombinedTrend_Breakout_VolumeFilter
# Hypothesis: Combine 12h EMA trend with 4h Donchian breakout and volume confirmation to capture strong trends while avoiding whipsaws.
# Works in bull markets by buying breakouts above bullish trend; in bear markets by selling breakouts below bearish trend.
# Volume filter ensures institutional participation. Target: 20-40 trades/year to minimize fee drag.

name = "4h_12h_CombinedTrend_Breakout_VolumeFilter"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h EMA34 for trend
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 4h Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > (vol_ema20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(high_20[i]) or np.isnan(low_20[i]) or
            np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high + 12h EMA uptrend + volume
            if close[i] > high_20[i] and close[i] > ema_12h_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + 12h EMA downtrend + volume
            elif close[i] < low_20[i] and close[i] < ema_12h_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below Donchian low or trend changes
            if close[i] < low_20[i] or close[i] < ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above Donchian high or trend changes
            if close[i] > high_20[i] or close[i] > ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals