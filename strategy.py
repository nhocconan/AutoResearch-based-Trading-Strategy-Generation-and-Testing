#!/usr/bin/env python3
"""
1h_4H_Donchian_1D_EMA_Filter
Hypothesis: 1-hour Donchian breakouts filtered by 4-hour EMA trend and 1-day EMA for stronger trend alignment.
Uses 4h EMA for direction (trend filter) and 1-day EMA for stronger bias, with volume confirmation.
Designed for low trade frequency (15-30/year) with clear entry/exit rules to minimize fee drag.
Works in both bull and bear markets by following the dominant trend on higher timeframes.
"""

name = "1h_4H_Donchian_1D_EMA_Filter"
timeframe = "1h"
leverage = 1.0

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
    
    # Calculate Donchian Channel (20-period) on 1h
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Get 4-hour EMA (50) for trend direction
    df_4h = get_htf_data(prices, '4h')
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1-day EMA (50) for stronger trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if position == 0:
            # LONG: Price breaks above Donchian upper with 4h/1d EMA uptrend and volume
            if close[i] > donchian_upper[i] and close[i] > ema_50_4h_aligned[i] and close[i] > ema_50_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below Donchian lower with 4h/1d EMA downtrend and volume
            elif close[i] < donchian_lower[i] and close[i] < ema_50_4h_aligned[i] and close[i] < ema_50_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian lower or 4h EMA turns down
            if close[i] < donchian_lower[i] or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian upper or 4h EMA turns up
            if close[i] > donchian_upper[i] or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals