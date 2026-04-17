#!/usr/bin/env python3
"""
12h_PriceChannel_Breakout_v1
Breakout from 20-period Donchian channel on 12h with volume confirmation and 1d trend filter.
Designed to capture strong moves with minimal trades in both bull and bear markets.
Target: 50-150 total trades over 4 years (12-37/year).
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
    
    # === 12h Donchian Channel (20-period) ===
    # Calculate on 12h data then align
    df_12h = get_htf_data(prices, '12h')
    donch_high_12h = pd.Series(df_12h['high'].values).rolling(window=20, min_periods=20).max().values
    donch_low_12h = pd.Series(df_12h['low'].values).rolling(window=20, min_periods=20).min().values
    donch_high_12h_aligned = align_htf_to_ltf(prices, df_12h, donch_high_12h)
    donch_low_12h_aligned = align_htf_to_ltf(prices, df_12h, donch_low_12h)
    
    # === 1d EMA50 for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Volume confirmation (12h volume > 1.5x 20-period average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 60
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high_12h_aligned[i]) or 
            np.isnan(donch_low_12h_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: break above Donchian high + volume + above 1d EMA50
            if (close[i] > donch_high_12h_aligned[i] and 
                volume[i] > 1.5 * vol_ma_20[i] and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: break below Donchian low + volume + below 1d EMA50
            elif (close[i] < donch_low_12h_aligned[i] and 
                  volume[i] > 1.5 * vol_ma_20[i] and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: reverse signal or volatility-based exit
        elif position == 1:
            # Exit long: reverse signal OR price retrace to midpoint
            midpoint = (donch_high_12h_aligned[i] + donch_low_12h_aligned[i]) / 2
            if (close[i] < donch_low_12h_aligned[i] or 
                close[i] < midpoint):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: reverse signal OR price retrace to midpoint
            midpoint = (donch_high_12h_aligned[i] + donch_low_12h_aligned[i]) / 2
            if (close[i] > donch_high_12h_aligned[i] or 
                close[i] > midpoint):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_PriceChannel_Breakout_v1"
timeframe = "12h"
leverage = 1.0