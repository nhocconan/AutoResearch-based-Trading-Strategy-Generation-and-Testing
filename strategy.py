#!/usr/bin/env python3
"""
4h_Donchian_Breakout_Trend_Volume_Spike_v1
Hypothesis: Breakouts above/below Donchian(20) channel with trend filter (price > SMA50 for longs, < SMA50 for shorts) and volume spike (>1.5x 20-period average) on 4h timeframe. Designed to capture strong momentum moves while avoiding choppy markets. Trend filter ensures alignment with medium-term direction, reducing false breakouts in sideways markets. Volume spike confirms institutional participation. Designed for low trade frequency (<50/year) to minimize fee drag and improve generalization in both bull and bear markets.
"""

name = "4h_Donchian_Breakout_Trend_Volume_Spike_v1"
timeframe = "4h"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d SMA50 for trend filter
    sma_50_1d = pd.Series(df_1d['close']).rolling(window=50, min_periods=50).mean().values
    sma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_50_1d)
    
    # Donchian channel on 4h: 20-period high/low
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if position == 0:
            # LONG: Price breaks above Donchian high with volume spike and uptrend
            if not np.isnan(donchian_high[i]) and high[i] > donchian_high[i] and \
               volume_spike[i] and close[i] > sma_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low with volume spike and downtrend
            elif not np.isnan(donchian_low[i]) and low[i] < donchian_low[i] and \
                 volume_spike[i] and close[i] < sma_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian low or trend weakens
            if not np.isnan(donchian_low[i]) and low[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian high or trend weakens
            if not np.isnan(donchian_high[i]) and high[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals