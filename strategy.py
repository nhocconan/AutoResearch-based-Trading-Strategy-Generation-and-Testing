#!/usr/bin/env python3
# 1H_4H_1D_Trend_Structure
# Hypothesis: On 1h timeframe, use 4h Supertrend for trend direction and 1d Donchian breakout for entry timing.
# Long when: 4h Supertrend = up, price breaks above 1d upper Donchian (20), volume > 20-period MA.
# Short when: 4h Supertrend = down, price breaks below 1d lower Donchian (20), volume > 20-period MA.
# Exit when: 4h Supertrend reverses.
# Uses 4h Supertrend for trend filter and 1d Donchian for structure to reduce whipsaw.
# Targets 15-30 trades/year for low fee drag on 1h timeframe.

name = "1H_4H_1D_Trend_Structure"
timeframe = "1h"
leverage = 1.0

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
    
    # Load 4h data for Supertrend calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Supertrend (ATR=10, multiplier=3.0)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = np.abs(high_4h[1:] - low_4h[1:])
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with indices
    
    # ATR(10)
    atr_10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Basic Upper and Lower Bands
    basic_ub = (high_4h + low_4h) / 2 + 3.0 * atr_10
    basic_lb = (high_4h + low_4h) / 2 - 3.0 * atr_10
    
    # Final Upper and Lower Bands
    final_ub = np.zeros_like(close_4h)
    final_lb = np.zeros_like(close_4h)
    for i in range(len(close_4h)):
        if i == 0:
            final_ub[i] = basic_ub[i]
            final_lb[i] = basic_lb[i]
        else:
            if close_4h[i-1] <= final_ub[i-1]:
                final_ub[i] = min(basic_ub[i], final_ub[i-1])
            else:
                final_ub[i] = basic_ub[i]
            if close_4h[i-1] >= final_lb[i-1]:
                final_lb[i] = max(basic_lb[i], final_lb[i-1])
            else:
                final_lb[i] = basic_lb[i]
    
    # Supertrend
    supertrend = np.zeros_like(close_4h)
    for i in range(len(close_4h)):
        if i == 0:
            supertrend[i] = 1.0  # start with uptrend
        else:
            if close_4h[i] <= final_ub[i]:
                supertrend[i] = 1.0  # uptrend
            else:
                supertrend[i] = -1.0  # downtrend
    
    # Load 1d data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align indicators to 1h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_4h, supertrend)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Volume confirmation: 20-period moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(supertrend_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        st = supertrend_aligned[i]
        dch = donchian_high_aligned[i]
        dcl = donchian_low_aligned[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # LONG: 4h uptrend, price breaks above 1d upper Donchian, volume > 20MA
            if st == 1.0 and close[i] > dch and volume[i] > vol_ma_val:
                signals[i] = 0.20
                position = 1
            # SHORT: 4h downtrend, price breaks below 1d lower Donchian, volume > 20MA
            elif st == -1.0 and close[i] < dcl and volume[i] > vol_ma_val:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: 4h trend reverses to down
            if st == -1.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: 4h trend reverses to up
            if st == 1.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals