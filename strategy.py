#!/usr/bin/env python3
"""
4h_Donchian_Breakout_With_Volume_And_Trend_v1
Long when price breaks above Donchian(20) high, short when breaks below Donchian(20) low.
Trend filter: price above/below 1d EMA200.
Volume filter: current volume > 1.5x 20-period average volume.
Exit when price crosses back through the Donchian midpoint.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Donchian Channel (20) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # === Volume Filter ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === 1d EMA200 for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    ema_200_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 200
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(ema_200_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: break above Donchian high + volume filter + price above 1d EMA200
            if (close[i] > donchian_high[i] and 
                volume[i] > 1.5 * vol_ma[i] and 
                close[i] > ema_200_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: break below Donchian low + volume filter + price below 1d EMA200
            elif (close[i] < donchian_low[i] and 
                  volume[i] > 1.5 * vol_ma[i] and 
                  close[i] < ema_200_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price crosses below Donchian midpoint
            if close[i] < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above Donchian midpoint
            if close[i] > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_With_Volume_And_Trend_v1"
timeframe = "4h"
leverage = 1.0