#!/usr/bin/env python3
"""
Hypothesis: 12-hour strategy using 1-day Donchian breakout with volume confirmation and 1-day trend filter.
Long when price breaks above 1-day high + price above 1-day EMA20 + volume surge.
Short when price breaks below 1-day low + price below 1-day EMA20 + volume surge.
Exit when price returns to 1-day midpoint or trend reverses.
Designed for low turnover: ~15-30 trades/year per symbol to minimize fee drag.
Works in bull markets via breakouts and in bear via short-side symmetry.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data once for Donchian channels and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1-day Donchian channels (20)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donch_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # 1-day EMA20 for trend filter
    close_1d = df_1d['close'].values
    ema_20 = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume filter: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(30, n):
        # 1-day index
        idx_1d = i // 2  # 2 bars per day (12h timeframe)
        if idx_1d < 20:  # need enough for Donchian/EMA
            continue
        
        # Get previous 1-day values to avoid look-ahead
        high_prev = donch_high[idx_1d - 1] if idx_1d - 1 < len(donch_high) else donch_high[-1]
        low_prev = donch_low[idx_1d - 1] if idx_1d - 1 < len(donch_low) else donch_low[-1]
        mid_prev = donch_mid[idx_1d - 1] if idx_1d - 1 < len(donch_mid) else donch_mid[-1]
        ema20_prev = ema_20[idx_1d - 1] if idx_1d - 1 < len(ema_20) else ema_20[-1]
        if np.isnan(high_prev) or np.isnan(low_prev) or np.isnan(mid_prev) or np.isnan(ema20_prev):
            continue
        
        # Create arrays for alignment (using previous values)
        high_arr = np.full(len(df_1d), high_prev)
        low_arr = np.full(len(df_1d), low_prev)
        mid_arr = np.full(len(df_1d), mid_prev)
        ema20_arr = np.full(len(df_1d), ema20_prev)
        high_12h = align_htf_to_ltf(prices, df_1d, high_arr)[i]
        low_12h = align_htf_to_ltf(prices, df_1d, low_arr)[i]
        mid_12h = align_htf_to_ltf(prices, df_1d, mid_arr)[i]
        ema20_12h = align_htf_to_ltf(prices, df_1d, ema20_arr)[i]
        
        if position == 0:
            # Long: price breaks above 1d high + price above EMA20 + volume surge
            if (close[i] > high_12h and 
                close[i] > ema20_12h and 
                volume[i] > vol_ma[i] * 2.0):
                position = 1
                signals[i] = position_size
            # Short: price breaks below 1d low + price below EMA20 + volume surge
            elif (close[i] < low_12h and 
                  close[i] < ema20_12h and 
                  volume[i] > vol_ma[i] * 2.0):
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: price returns to 1d mid or price below EMA20
            if close[i] < mid_12h or close[i] < ema20_12h:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: price returns to 1d mid or price above EMA20
            if close[i] > mid_12h or close[i] > ema20_12h:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "12h_1d_Donchian_EMA20_Volume"
timeframe = "12h"
leverage = 1.0