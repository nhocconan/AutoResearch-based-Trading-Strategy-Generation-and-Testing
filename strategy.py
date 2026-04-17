#!/usr/bin/env python3
"""
12h_Donchian20_Volume_Trend_v1
Donchian(20) breakout + volume confirmation + 1d EMA50 trend filter.
Long: price breaks above upper Donchian + volume > 1.5x avg + price > 1d EMA50.
Short: price breaks below lower Donchian + volume > 1.5x avg + price < 1d EMA50.
Exit: price re-enters Donchian channel or trend filter fails.
Designed to capture strong trends with volume confirmation.
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
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Volume Average (20) ===
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === 1d EMA50 for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(vol_avg[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price > upper Donchian, volume > 1.5x avg, price > 1d EMA50
            if (close[i] > highest_high[i] and 
                volume[i] > 1.5 * vol_avg[i] and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price < lower Donchian, volume > 1.5x avg, price < 1d EMA50
            elif (close[i] < lowest_low[i] and 
                  volume[i] > 1.5 * vol_avg[i] and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price < lower Donchian OR price < 1d EMA50
            if (close[i] < lowest_low[i] or 
                close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > upper Donchian OR price > 1d EMA50
            if (close[i] > highest_high[i] or 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Volume_Trend_v1"
timeframe = "12h"
leverage = 1.0