#!/usr/bin/env python3
"""
12h_1d_Weekly_Trend_Filter
Hypothesis: Uses weekly trend direction (EMA200) with 12h Donchian breakout and volume confirmation.
Trades only in direction of weekly trend to avoid counter-trend whipsaws. Works in bull (follow trend) and bear (counter-trend bounces against long-term weekly downtrend).
Target: 15-30 trades/year on 12h (60-120 total over 4 years).
"""

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
    
    # Get weekly data for trend filter (EMA200)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Get daily data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian channels (20-period) on daily
    donch_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align weekly EMA200 and daily Donchian to 12h timeframe
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    donch_high_20_aligned = align_htf_to_ltf(prices, df_1d, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_1d, donch_low_20)
    
    # Volume confirmation on 12h
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema200_1w_aligned[i]) or 
            np.isnan(donch_high_20_aligned[i]) or 
            np.isnan(donch_low_20_aligned[i]) or
            np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from weekly EMA200
        weekly_uptrend = close[i] > ema200_1w_aligned[i]
        weekly_downtrend = close[i] < ema200_1w_aligned[i]
        
        # Long conditions: weekly uptrend + price breaks above Donchian high + volume expansion
        if weekly_uptrend and close[i] > donch_high_20_aligned[i] and volume_expansion[i]:
            if position != 1:
                position = 1
                signals[i] = position_size
            else:
                signals[i] = position_size
        
        # Short conditions: weekly downtrend + price breaks below Donchian low + volume expansion
        elif weekly_downtrend and close[i] < donch_low_20_aligned[i] and volume_expansion[i]:
            if position != -1:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = -position_size
        
        # Exit conditions: reverse signal or loss of momentum
        elif position == 1 and (close[i] < donch_low_20_aligned[i] or not weekly_uptrend):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > donch_high_20_aligned[i] or not weekly_downtrend):
            position = 0
            signals[i] = 0.0
        elif position == 1:
            signals[i] = position_size
        elif position == -1:
            signals[i] = -position_size
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_1d_Weekly_Trend_Filter"
timeframe = "12h"
leverage = 1.0