#!/usr/bin/env python3
"""
1h_4h_DonchianBreakout_v1
Hypothesis: Use 4h Donchian channel breakouts with volume confirmation for direction,
and 1h for precise entry timing. Long when price breaks above 4h Donchian high (20) with
volume > 1.5x 20-bar average. Short when price breaks below 4h Donchian low (20) with
volume > 1.5x 20-bar average. Exit when price crosses back through the 4h Donchian middle.
Designed for 1h timeframe to reduce trade frequency vs lower timeframes while capturing
trend moves. Volume filter reduces false breakouts. Works in bull markets by buying
breakouts and in bear markets by selling breakdowns.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data once for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # 4h Donchian channel (20-period)
    high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    mid_20 = (high_20 + low_20) / 2.0
    
    # Align to 1h timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_4h, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_4h, low_20)
    mid_20_aligned = align_htf_to_ltf(prices, df_4h, mid_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or
            np.isnan(mid_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.5 * vol_ma
        else:
            volume_ok = False
        
        if position == 0:
            # Long conditions: break above 4h Donchian high + volume confirmation
            if price > high_20_aligned[i] and volume_ok:
                signals[i] = 0.20
                position = 1
            # Short conditions: break below 4h Donchian low + volume confirmation
            elif price < low_20_aligned[i] and volume_ok:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back below 4h Donchian middle
            if price < mid_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price crosses back above 4h Donchian middle
            if price > mid_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4h_DonchianBreakout_v1"
timeframe = "1h"
leverage = 1.0