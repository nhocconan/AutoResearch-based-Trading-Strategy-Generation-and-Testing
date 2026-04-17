#!/usr/bin/env python3
"""
12h_1w_Donchian_Breakout_Retest_v1
Breakout of weekly Donchian channel (20) with retest of breakout level.
Trades only in direction of 1d EMA200 trend filter.
Exit on opposite Donchian breakout or close below/above 1d EMA50.
Designed for low-frequency, high-conviction trades in both bull and bear markets.
Target: 50-150 total trades over 4 years (12-37/year).
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
    
    # === Weekly Donchian Channel (20) ===
    df_1w = get_htf_data(prices, '1w')
    dh_20 = pd.Series(df_1w['high'].values).rolling(window=20, min_periods=20).max().values
    dl_20 = pd.Series(df_1w['low'].values).rolling(window=20, min_periods=20).min().values
    dh_20_aligned = align_htf_to_ltf(prices, df_1w, dh_20)
    dl_20_aligned = align_htf_to_ltf(prices, df_1w, dl_20)
    
    # === Daily EMAs for trend and exit filters ===
    df_1d = get_htf_data(prices, '1d')
    ema_200_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 200
    
    # Track position state and breakout levels
    position = 0  # 0: flat, 1: long, -1: short
    long_breakout_level = 0.0
    short_breakout_level = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(dh_20_aligned[i]) or 
            np.isnan(dl_20_aligned[i]) or 
            np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: break above weekly Donchian high, retest, and above 1d EMA200
            if (high[i] > dh_20_aligned[i] and 
                low[i] <= dh_20_aligned[i] * 1.001 and  # retest within 0.1%
                close[i] > ema_200_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
                long_breakout_level = dh_20_aligned[i]
                continue
            # Short: break below weekly Donchian low, retest, and below 1d EMA200
            elif (low[i] < dl_20_aligned[i] and 
                  high[i] >= dl_20_aligned[i] * 0.999 and  # retest within 0.1%
                  close[i] < ema_200_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
                short_breakout_level = dl_20_aligned[i]
                continue
        
        # Exit logic for long
        elif position == 1:
            # Exit: break below weekly Donchian low OR close below 1d EMA50
            if (low[i] < dl_20_aligned[i] or 
                close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        # Exit logic for short
        elif position == -1:
            # Exit: break above weekly Donchian high OR close above 1d EMA50
            if (high[i] > dh_20_aligned[i] or 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1w_Donchian_Breakout_Retest_v1"
timeframe = "12h"
leverage = 1.0