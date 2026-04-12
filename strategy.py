#!/usr/bin/env python3
"""
6h_1w_donchian_volume_v1
Hypothesis: 6-hour Donchian breakout with weekly trend filter and volume confirmation.
- Long: price breaks above 20-period Donchian high AND price > weekly EMA200 (bullish trend) AND volume spike
- Short: price breaks below 20-period Donchian low AND price < weekly EMA200 (bearish trend) AND volume spike
- Uses weekly trend to avoid counter-trend trades in strong trends, works in both bull/bear markets.
Target: 15-35 trades/year (60-140 total over 4 years) to minimize fee drag.
"""

name = "6h_1w_donchian_volume_v1"
timeframe = "6h"
leverage = 1.0

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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Weekly EMA200 for trend direction
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # 6h Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(ema200_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: price breaks above Donchian high AND weekly uptrend AND volume spike
        if (close[i] > high_max[i] and close[i] > ema200_1w_aligned[i] and vol_confirm[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: price breaks below Donchian low AND weekly downtrend AND volume spike
        elif (close[i] < low_min[i] and close[i] < ema200_1w_aligned[i] and vol_confirm[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: reverse signal or price crosses back to opposite Donchian level
        elif position == 1 and close[i] < low_min[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > high_max[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals