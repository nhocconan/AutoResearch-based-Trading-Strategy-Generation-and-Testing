#!/usr/bin/env python3
"""
6h_1d_Donchian_Weekly_Direction
Hypothesis: Use 1-day Donchian breakout with weekly directional filter (price above/below weekly 200 EMA).
In bull markets, buy breakouts above 1-day high; in bear markets, sell breakdowns below 1-day low.
Volume confirmation filters weak breakouts. Weekly trend ensures alignment with higher timeframe momentum.
Target: 15-30 trades/year.
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
    
    # Get daily data for Donchian levels and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1-day Donchian channels (20-period)
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Weekly trend filter: price above/below weekly 200 EMA
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    weekly_uptrend = close_1w > ema_200_1w
    weekly_downtrend = close_1w < ema_200_1w
    
    # Volume confirmation: current 1d volume > 1.5x 20-period average
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_expansion = volume_1d > (vol_ma_20_1d * 1.5)
    
    # Align all data to 6h timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend.astype(float))
    volume_expansion_aligned = align_htf_to_ltf(prices, df_1d, volume_expansion.astype(float))
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or
            np.isnan(weekly_uptrend_aligned[i]) or np.isnan(weekly_downtrend_aligned[i]) or
            np.isnan(volume_expansion_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long: breakout above 1-day high in weekly uptrend with volume expansion
        long_condition = (close[i] > high_20_aligned[i]) and weekly_uptrend_aligned[i] > 0.5 and volume_expansion_aligned[i]
        
        # Short: breakdown below 1-day low in weekly downtrend with volume expansion
        short_condition = (close[i] < low_20_aligned[i]) and weekly_downtrend_aligned[i] > 0.5 and volume_expansion_aligned[i]
        
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif long_condition and position == 1:
            signals[i] = position_size
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        elif short_condition and position == -1:
            signals[i] = -position_size
        else:
            # Exit conditions: reverse signal or loss of trend/volume
            if position == 1 and (not weekly_uptrend_aligned[i] > 0.5 or not volume_expansion_aligned[i]):
                position = 0
                signals[i] = 0.0
            elif position == -1 and (not weekly_downtrend_aligned[i] > 0.5 or not volume_expansion_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "6h_1d_Donchian_Weekly_Direction"
timeframe = "6h"
leverage = 1.0