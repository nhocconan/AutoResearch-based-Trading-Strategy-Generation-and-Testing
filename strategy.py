#!/usr/bin/env python3
"""
6h_LongTermBreakout_1dTrend_Volume
Hypothesis: Breakouts above 1d Donchian(20) high/low on 6h timeframe with volume confirmation and 1d EMA50 trend filter. Designed for 6h to capture longer-term trends in both bull and bear markets with limited trade frequency to reduce fee drag.
"""

name = "6h_LongTermBreakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

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
    
    # Get 1d data for Donchian channels and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Donchian(20) channels
    donchian_high = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 1.5x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_confirmed = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # LONG: Break above Donchian high with volume confirmation in uptrend
        if position == 0 and \
           donchian_high_aligned[i] > 0 and not np.isnan(donchian_high_aligned[i]) and \
           high[i] > donchian_high_aligned[i] and volume_confirmed[i] and \
           close[i] > ema_50_1d_aligned[i]:
            signals[i] = 0.25
            position = 1
        # SHORT: Break below Donchian low with volume confirmation in downtrend
        elif position == 0 and \
             donchian_low_aligned[i] > 0 and not np.isnan(donchian_low_aligned[i]) and \
             low[i] < donchian_low_aligned[i] and volume_confirmed[i] and \
             close[i] < ema_50_1d_aligned[i]:
            signals[i] = -0.25
            position = -1
        # EXIT LONG: Price breaks below Donchian low or trend weakens
        elif position == 1 and (
            donchian_low_aligned[i] > 0 and not np.isnan(donchian_low_aligned[i]) and
            low[i] < donchian_low_aligned[i]
        ) or (position == 1 and close[i] < ema_50_1d_aligned[i]):
            signals[i] = 0.0
            position = 0
        # EXIT SHORT: Price breaks above Donchian high or trend weakens
        elif position == -1 and (
            donchian_high_aligned[i] > 0 and not np.isnan(donchian_high_aligned[i]) and
            high[i] > donchian_high_aligned[i]
        ) or (position == -1 and close[i] > ema_50_1d_aligned[i]):
            signals[i] = 0.0
            position = 0
        else:
            # Maintain current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals