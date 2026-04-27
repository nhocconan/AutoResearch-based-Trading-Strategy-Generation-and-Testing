#!/usr/bin/env python3
"""
#100740 - 4h_Donchian20_1dTrend_VolumeBreakout
Hypothesis: 4h Donchian breakout with 1d EMA trend filter and volume confirmation.
Works in bull (breakouts with trend) and bear (mean reversion to mean) by using trend filter to select direction.
Targets 20-50 trades/year to minimize fee drag. Uses 4h primary timeframe with 1d HTF for trend and volume filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 1d volume average for volume filter
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # Calculate Donchian channels on 4h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_avg_1d_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: price breaks above Donchian high, above 1d EMA50, volume > 1.5x average
        if (close[i] > donchian_high[i] and 
            close[i] > ema50_1d_aligned[i] and 
            volume[i] > (vol_avg_1d_aligned[i] * 1.5)):
            signals[i] = 0.25
            position = 1
        # Short condition: price breaks below Donchian low, below 1d EMA50, volume > 1.5x average
        elif (close[i] < donchian_low[i] and 
              close[i] < ema50_1d_aligned[i] and 
              volume[i] > (vol_avg_1d_aligned[i] * 1.5)):
            signals[i] = -0.25
            position = -1
        # Exit conditions: price returns to opposite Donchian level
        elif position == 1 and close[i] < donchian_low[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > donchian_high[i]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_1dTrend_VolumeBreakout"
timeframe = "4h"
leverage = 1.0