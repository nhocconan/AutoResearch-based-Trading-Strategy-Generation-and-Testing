#!/usr/bin/env python3
"""
#100743 - 4h_PriceChannel_1dTrend_VolumeFilter
Hypothesis: Price channel breakout with 1d trend filter and volume confirmation works in both bull and bear markets.
In bull: breaks above upper channel with up trend = momentum continuation.
In bear: breaks below lower channel with down trend = continuation of downtrend.
Uses 4h price channels (Donchian-like) with 1d EMA50 trend filter and volume spike.
Targets 20-40 trades/year to minimize fee drag.
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 40-period high/low for price channel (4h timeframe)
    high_max = pd.Series(high).rolling(window=40, min_periods=40).max().values
    low_min = pd.Series(low).rolling(window=40, min_periods=40).min().values
    
    # Volume filter: volume > 1.8x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(high_max[i]) or 
            np.isnan(low_min[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: price breaks above 40-period high, above 1d EMA50, volume spike
        if (close[i] > high_max[i] and 
            close[i] > ema50_1d_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short condition: price breaks below 40-period low, below 1d EMA50, volume spike
        elif (close[i] < low_min[i] and 
              close[i] < ema50_1d_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: price returns to opposite channel boundary (mean reversion)
        elif position == 1 and close[i] < low_min[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > high_max[i]:
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

name = "4h_PriceChannel_1dTrend_VolumeFilter"
timeframe = "4h"
leverage = 1.0