#!/usr/bin/env python3
"""
#100841 - 4h_Donchian20_Breakout_1wTrend_VolumeSpike
Hypothesis: Donchian breakout with weekly trend filter and volume spike. Works in bull (breakouts with trend) and bear (breakouts against trend). Targets 20-40 trades/year to minimize fee drag.
"""

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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate Donchian channels (20-period) for breakout levels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 2.0x 50-period average (stricter to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: price breaks above Donchian high, above weekly EMA50, volume spike
        if (close[i] > donchian_high[i] and 
            close[i] > ema50_1w_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.30
            position = 1
        # Short condition: price breaks below Donchian low, below weekly EMA50, volume spike
        elif (close[i] < donchian_low[i] and 
              close[i] < ema50_1w_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.30
            position = -1
        # Exit conditions: price returns to opposite Donchian level (mean reversion)
        elif position == 1 and close[i] < donchian_low[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > donchian_high[i]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.30
            elif position == -1:
                signals[i] = -0.30
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_Breakout_1wTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0