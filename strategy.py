#!/usr/bin/env python3
"""
12h_1D_4H_Trend_Filter
Hypothesis: On 12h timeframe, use 1D EMA34 for trend direction and 4H Donchian breakout for entry timing.
Price breaks above/below Donchian(20) on 4H only when aligned with 1D trend. Volume confirmation filters false signals.
Designed for low trade frequency (12-25/year) to work in both bull and bear markets by capturing trend continuations
with proper filtering. Uses 1D trend filter to avoid whipsaws in ranging markets.
"""

name = "12h_1D_4H_Trend_Filter"
timeframe = "12h"
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
    
    # Get 1D data for trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    ema_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Get 4H data for Donchian channel
    df_4h = get_htf_data(prices, '4h')
    donch_high = pd.Series(df_4h['high']).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(df_4h['low']).rolling(window=20, min_periods=20).min().values
    donch_high_aligned = align_htf_to_ltf(prices, df_4h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_4h, donch_low)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):
        # Determine trend direction from 1D EMA
        uptrend = close[i] > ema_1d_aligned[i]
        downtrend = close[i] < ema_1d_aligned[i]
        
        if position == 0:
            # LONG: Uptrend + price breaks above Donchian high + volume
            if uptrend and close[i] > donch_high_aligned[i] and close[i-1] <= donch_high_aligned[i-1] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Downtrend + price breaks below Donchian low + volume
            elif downtrend and close[i] < donch_low_aligned[i] and close[i-1] >= donch_low_aligned[i-1] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian low or trend turns down
            if close[i] < donch_low_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian high or trend turns up
            if close[i] > donch_high_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals